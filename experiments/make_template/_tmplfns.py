from __future__ import annotations

import itertools as itrtls
import math
from typing import Callable, Optional

import lightning as pl
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import scipy.spatial as sp_spatial
import tensordict as thd
import torch as th
import torch.distributions.utils
import torchmetrics as thm
import tqdm.auto as tqdm


def feature_masks_to_feature_combs(tmpls: th.Tensor):
    return [tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls]


# NOTE identify initial feature
def make_feature_masks(
    n_covs: int,
    n_masks: int,
    min_features: int,
    max_features: Optional[int],
) -> th.Tensor:
    max_features = n_covs if max_features is None else max_features
    bincnt_fcs_l: list[int] = [
        # in order to accomondate for init_fidx,
        # both n_covs and i is one less than desired n_feats
        min(math.comb(n_covs, i), th.iinfo(th.long).max)
        for i in range(min_features, max_features + 1)
    ]
    n_masks = min(n_masks, sum(bincnt_fcs_l))
    bincnt_fcs: th.Tensor = th.as_tensor(bincnt_fcs_l, dtype=th.long)
    ps: th.Tensor = th.ones_like(bincnt_fcs, dtype=th.float32)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_masks, replacement=True), minlength=len(bincnt_fcs)
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(_curr_bincnts >= bincnt_fcs, 0.0, 1.0)
        _realloc_cnts: th.Tensor = th.where(
            _curr_bincnts > bincnt_fcs, _curr_bincnts - bincnt_fcs, 0
        )
        _tmp_bincnts: th.Tensor = th.bincount(
            th.multinomial(
                _tmp_ps,
                int(th.sum(_realloc_cnts).item()),
                replacement=True,
            ),
            minlength=len(bincnt_fcs),
        )
        _curr_bincnts = _curr_bincnts - _realloc_cnts + _tmp_bincnts
    nfc_from_each_binned_fcs = _curr_bincnts
    # make unique feature combination
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [set() for _ in bincnt_fcs]
    for _k, (_count, _fcs_set) in enumerate(
        zip(nfc_from_each_binned_fcs, fcs_sets_by_bins)
    ):
        if _count == 0:
            continue
        _nfeats: int = _k + min_features
        while len(_fcs_set) < _count:
            _fc_l: list[int] = th.multinomial(
                th.ones((n_covs,)), num_samples=_nfeats
            ).tolist()
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc = tuple(_fc_l)
            if _fc not in _fcs_set:
                _fcs_set.add(_fc)
    # from fcomb to act
    fms: th.Tensor = th.zeros((n_masks, n_covs), dtype=th.long)
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    assert len(fms) == len(fcs_l)
    for _i, _fc in enumerate(fcs_l):
        fms[_i, _fc] = 1
    return fms


def _ident_init_fidx_single(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    max_features: Optional[int],
    n_iter: int,
    lmbda: float,
    bsz: int,
) -> tuple[int, th.Tensor]:
    n_covs: int = tdata["xs"].shape[1]
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(
            make_feature_masks(
                n_covs=n_covs,
                n_masks=len(tdata),
                min_features=1,
                max_features=max_features,
            )
        )
    fms: th.Tensor = make_feature_masks(
        n_covs=n_covs, n_masks=n_iter, min_features=1, max_features=max_features
    )
    best_mask_cost: th.Tensor = th.inf * th.ones((n_covs,))
    n_iter = min(n_iter, len(fms))
    for _i in tqdm.trange(
        n_iter, desc="ident_init_fidx_single", leave=False, dynamic_ncols=True
    ):
        # (n_covs, )
        _bfm: th.Tensor = fms[_i]
        # (bsz, )
        _btidxs: th.Tensor = th.randint(0, len(tdata), (bsz,))
        # (bsz, n_covs)
        _btdata: thd.TensorDict = tdata[_btidxs]
        _bfms: th.Tensor = _bfm[None, :].expand(bsz, -1)
        # (bsz, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(_btdata["xs"], _bfms)
        _blyhats: th.Tensor = torch.distributions.utils.probs_to_logits(_bpyhats)
        # ()
        _bcosts: th.Tensor = th.nn.functional.cross_entropy(
            _blyhats, _btdata["ys"]
        ) + lmbda * th.sum(_bfm)
        best_mask_cost = th.minimum(
            best_mask_cost, th.where(_bfm == 1, _bcosts, th.inf)
        )
    return int(th.argmin(best_mask_cost).item()), best_mask_cost


def identify_init_fidx(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    max_features: Optional[int],
    n_repeat: int,
    n_iter: int,
    lmbda: float,
    bsz: int,
) -> tuple[int, th.Tensor]:
    best_fms: th.Tensor = th.stack(
        [
            _ident_init_fidx_single(
                tdata=tdata,
                classifier=classifier,
                max_features=max_features,
                n_iter=n_iter,
                lmbda=lmbda,
                bsz=bsz,
            )[1]
            for _ in tqdm.trange(
                n_repeat, desc="ident init_fidx", leave=False, dynamic_ncols=True
            )
        ]
    )
    best_fm: th.Tensor = th.mean(best_fms, dim=0)
    return int(th.argmin(best_fm)), best_fm


# NOTE make candidates
def make_template_candidates(
    n_covs: int,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
) -> th.Tensor:
    bincnt_fcs_l: list[int] = [
        # in order to accomondate for init_fidx,
        # both n_covs and i is one less than desired n_feats
        min(math.comb(n_covs - 1, i), th.iinfo(th.long).max)
        for i in range(
            min_features - 1, n_covs if max_features is None else max_features
        )
    ]
    n_cands: int = min(n_cands_targ, sum(bincnt_fcs_l))
    bincnt_fcs: th.Tensor = th.as_tensor(bincnt_fcs_l, dtype=th.long)
    ps: th.Tensor = th.ones_like(bincnt_fcs, dtype=th.float32)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_cands, replacement=True), minlength=len(bincnt_fcs)
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(_curr_bincnts >= bincnt_fcs, 0, 1.0)
        _realloc_cnts: th.Tensor = th.where(
            _curr_bincnts > bincnt_fcs, _curr_bincnts - bincnt_fcs, 0
        )
        _tmp_bincnts: th.Tensor = th.bincount(
            th.multinomial(
                _tmp_ps,
                int(th.sum(_realloc_cnts).item()),
                replacement=True,
            ),
            minlength=len(bincnt_fcs),
        )
        _curr_bincnts = _curr_bincnts - _realloc_cnts + _tmp_bincnts
    nfc_from_each_binned_fcs = _curr_bincnts
    # make unique feature combination
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [set() for _ in bincnt_fcs]
    for _k, (_count, _fcs_set) in enumerate(
        zip(nfc_from_each_binned_fcs, fcs_sets_by_bins)
    ):
        if _count == 0:
            continue
        _nfeats: int = _k + min_features
        while len(_fcs_set) < _count:
            _fc_l: list[int] = th.multinomial(
                th.ones((n_covs,)), num_samples=_nfeats
            ).tolist()
            # make sure initial feature is in fcomb
            if init_fidx not in _fc_l:
                _fc_l.append(init_fidx)
                _fc_l = _fc_l[1:]
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc = tuple(_fc_l)
            if _fc not in _fcs_set:
                _fcs_set.add(_fc)
    # from fcomb to act
    ctmpls: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    assert len(ctmpls) == len(fcs_l)
    for _i, _fc in enumerate(fcs_l):
        ctmpls[_i, _fc] = 1
    return ctmpls


def _mutate_tmpls(
    tmpls_prv: th.Tensor,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
) -> set[tuple[int, ...]]:
    # new candidate pool set
    fcs_set: set[tuple[int, ...]] = {
        tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls_prv
    }
    # previous template pool and exclude those that has no feature to mutate from
    tmpls_prv = tmpls_prv[th.sum(tmpls_prv, dim=1) - min_features > 0]
    # mutate templates
    pbar = tqdm.trange(
        n_cands_targ, desc="mutate tmpl_prv", dynamic_ncols=True, leave=False
    )
    for _ in pbar:
        if len(fcs_set) >= n_cands_targ:
            break
        # randomly choose an existing template to mutate from
        _tmpl_prv: th.Tensor = tmpls_prv[int(th.randint(0, len(tmpls_prv), ()).item())]
        # make copy of selected template
        _tmpl: th.Tensor = _tmpl_prv.clone()
        # always remove one feature from currently chosen templates
        _nfeats_mut: int = 1
        # indices to previously selected feature
        _fidxs: th.Tensor = th.argwhere(_tmpl == 1).flatten()
        # prevent init_fidx from being mutated
        _fidxs = _fidxs[_fidxs != init_fidx]
        # set mutated feature to be zero
        _tmpl[
            _fidxs[th.multinomial(th.ones_like(_fidxs, dtype=th.float64), _nfeats_mut)]
        ] = 0
        # add _tmpl to candidate pool
        _fc: tuple[int, ...] = tuple(th.argwhere(_tmpl == 1).flatten().tolist())
        if _fc not in fcs_set:
            fcs_set.add(_fc)
    pbar.close()
    return fcs_set


def _fill_fcs_set_with_random_tmpls(
    fcs_set: set[tuple[int, ...]],
    n_covs: int,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
    prv_featcounts: Optional[th.Tensor],
) -> list[set[tuple[int, ...]]]:
    # fill the rest of the demanded with randomly generated ones
    # group fcs_set by len(fcomb)
    max_features = n_covs if max_features is None else max_features
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [
        set() for _ in range(min_features - 1, max_features)
    ]
    [fcs_sets_by_bins[len(_fc) - min_features].add(_fc) for _fc in fcs_set]
    # compute maximum feature combinations allowed in each bin
    bincnt_fcs: th.Tensor = th.as_tensor(
        [
            # in order to accomondate for init_fidx,
            # both n_covs and i is one less than desired n_feats
            math.comb(n_covs - 1, i)
            for i in range(
                min_features - 1, n_covs if max_features is None else max_features
            )
        ],
        dtype=th.long,
    )
    n_cands_targ = (
        n_cands_targ if n_cands_targ <= th.sum(bincnt_fcs) else int(th.sum(bincnt_fcs))
    )
    if len(fcs_set) >= n_cands_targ:
        return fcs_sets_by_bins
    # subtract existing fcs from the bins
    bincnt_fcs = bincnt_fcs - th.as_tensor(
        list(map(len, fcs_sets_by_bins)), dtype=th.long
    )
    # sample number of fcs to add to existing feature combinations
    ps: th.Tensor = th.ones_like(bincnt_fcs, dtype=th.float32)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_cands_targ - len(fcs_set), replacement=True),
        minlength=len(bincnt_fcs),
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(_curr_bincnts >= bincnt_fcs, 0, 1.0)
        _realloc_cnts: th.Tensor = th.where(
            _curr_bincnts > bincnt_fcs, _curr_bincnts - bincnt_fcs, 0
        )
        _tmp_bincnts: th.Tensor = th.bincount(
            th.multinomial(
                _tmp_ps,
                int(th.sum(_realloc_cnts).item()),
                replacement=True,
            ),
            minlength=len(bincnt_fcs),
        )
        _curr_bincnts = _curr_bincnts - _realloc_cnts + _tmp_bincnts
    nfc_from_each_binned_fcs = _curr_bincnts
    # make unique feature combination
    for _k, (_count, _fcs_set) in enumerate(
        zip(nfc_from_each_binned_fcs, fcs_sets_by_bins)
    ):
        if _count == 0:
            continue
        _init_fcs_set_len: int = len(_fcs_set)
        _nfeats: int = _k + min_features
        while len(_fcs_set) - _init_fcs_set_len < _count:
            _ps: th.Tensor = (
                th.ones((n_covs,)) if prv_featcounts is None else prv_featcounts
            )
            _ps = th.where(_ps == 0, th.min(_ps[_ps > 0]), _ps)
            _ps = _ps.to(dtype=th.float32)
            _fc_l: list[int] = th.multinomial(_ps, num_samples=_nfeats).tolist()
            # make sure initial feature is in fcomb
            if init_fidx not in _fc_l:
                _fc_l.append(init_fidx)
                _fc_l = _fc_l[1:]
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc = tuple(_fc_l)
            if _fc not in _fcs_set:
                _fcs_set.add(_fc)
    return fcs_sets_by_bins


def update_template_candidates(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
) -> th.Tensor:
    fcs_set: set[tuple[int, ...]] = _mutate_tmpls(
        tmpls_prv=ctmpls[slctd_ms],
        init_fidx=init_fidx,
        n_cands_targ=n_cands_targ,
        min_features=min_features,
    )
    n_covs: int = ctmpls.shape[1]
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = _fill_fcs_set_with_random_tmpls(
        fcs_set=fcs_set,
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=n_cands_targ,
        min_features=min_features,
        max_features=max_features,
        prv_featcounts=(
            th.sum(ctmpls[slctd_ms], dim=0) if use_feature_importance_sampling else None
        ),
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = min(n_cands_targ, len(fcs_l))
    ctmpls_new: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    for _i, _fc in enumerate(fcs_l):
        ctmpls_new[_i, _fc] = 1
    return ctmpls_new


def update_template_candidates_fix_rounds(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    init_fidx: int,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
) -> th.Tensor:
    tmpls_prv: th.Tensor = ctmpls[slctd_ms]
    fcs_set: set[tuple[int, ...]] = _mutate_tmpls(
        tmpls_prv=tmpls_prv,
        init_fidx=init_fidx,
        n_cands_targ=2 * len(tmpls_prv),
        min_features=min_features,
    )
    n_covs: int = ctmpls.shape[1]
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = _fill_fcs_set_with_random_tmpls(
        fcs_set=fcs_set,
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=3 * len(tmpls_prv),
        min_features=min_features,
        max_features=max_features,
        prv_featcounts=(
            th.sum(ctmpls[slctd_ms], dim=0) if use_feature_importance_sampling else None
        ),
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = len(fcs_l)
    ctmpls_new: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    for _i, _fc in enumerate(fcs_l):
        ctmpls_new[_i, _fc] = 1
    return ctmpls_new


def precomp_rwds_for_tmpls(
    tmpls: th.Tensor,
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    lmbda: float,
    bsz: int,
) -> thd.TensorDict:
    txs: th.Tensor = data["xs"]
    tys: th.Tensor = data["ys"]
    n_cands: int = len(tmpls)
    n_labels: int = len(th.unique(tys))
    # (n_data,  n_cands, n_labels)
    pyhats: th.Tensor = th.empty((len(txs), n_cands, n_labels), dtype=th.float32)
    # (n_data,  n_cands)
    cels: th.Tensor = th.empty((len(txs), n_cands), dtype=th.float32)
    rwds: th.Tensor = th.empty_like(cels)
    pbar = tqdm.tqdm(
        th.split(th.arange(0, len(txs), dtype=th.long), bsz),
        desc="precomp candidates",
        leave=False,
        dynamic_ncols=True,
    )
    for _btidxs in pbar:
        _bsz: int = len(_btidxs)
        # (_bsz * n_cands, n_covs)
        _bctxs: th.Tensor = txs[_btidxs, None, :].expand(-1, n_cands, -1).flatten(0, 1)
        _bacts: th.Tensor = tmpls[None, :, :].expand(_bsz, -1, -1).flatten(0, 1)
        _bctxs = _bctxs * _bacts
        # (_bsz * n_cands, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(_bctxs, _bacts)
        _blyhats: th.Tensor = torch.distributions.utils.probs_to_logits(_bpyhats)
        # (_bsz * n_cands)
        _btys: th.Tensor = tys[_btidxs, None].expand(-1, n_cands).flatten(0, 1)
        _bcels: th.Tensor = th.nn.functional.cross_entropy(
            _blyhats, _btys, reduction="none"
        )
        _brwds: th.Tensor = -_bcels - lmbda * th.sum(_bacts, dim=1)
        pyhats[_btidxs] = _bpyhats.unflatten(0, (_bsz, n_cands))
        cels[_btidxs] = _bcels.unflatten(0, (_bsz, n_cands))
        rwds[_btidxs] = _brwds.unflatten(0, (_bsz, n_cands))
    pbar.close()
    # turn into tensordict
    tpcomp = thd.TensorDict(
        {"pyhats": pyhats, "cels": cels, "rwds": rwds}
    ).auto_batch_size_(1)
    return tpcomp


def make_templates_from_candidates(
    tpcomp: thd.TensorDict,
    ctmpls: th.Tensor,
    n_tmpls: int,
    plf: pl.Fabric,
    log_prefix: Optional[str] = "mk_tmpl",
) -> tuple[th.Tensor, th.Tensor]:
    # (n_data, n_cands)
    rwds: th.Tensor = tpcomp["rwds"]
    costs: th.Tensor = -rwds
    # template selected
    # **minimize cost**
    slctd_ms: th.Tensor = th.zeros((len(ctmpls)), dtype=th.bool)
    pbar = tqdm.trange(n_tmpls, desc="make templates", leave=False, dynamic_ncols=True)
    for _i in pbar:
        # start off with best template in the set
        if th.sum(slctd_ms) == 0:
            # (n_cands, )
            _fitns: th.Tensor = th.mean(costs, dim=0)
            _slctd: int = int(th.argmin(_fitns).item())
            slctd_ms[_slctd] = True
            metrics_d: dict[str, float] = {
                "cost": th.mean(th.min(costs[:, slctd_ms], dim=1)[0]).item(),
                "fitness": _fitns[_slctd].item(),
            }
            pbar.set_postfix(metrics_d)
            if log_prefix is not None:
                metrics_d = mylib.utils.add_prefix_to_dict(metrics_d, log_prefix)
            plf.log_dict(metrics_d, _i)
            continue
        # compute currently selected template set cost for each instance
        # (n_data, )
        _slctd_costs = th.min(costs[:, slctd_ms], dim=1)[0]
        # compute potential improvements of each available template for each instance
        # (n_data, n_avail_cands)
        _adj_costs: th.Tensor = costs[:, ~slctd_ms] - _slctd_costs[:, None]
        _adj_costs = th.where(_adj_costs < 0.0, _adj_costs, 0.0)
        # compute average improvements over all instances for each template
        # already selected template is masked out
        # (n_cands, )
        _fitns: th.Tensor = th.zeros((len(ctmpls),))
        _fitns[~slctd_ms] = th.mean(_adj_costs, dim=0)
        # terminate loop early if no improvement is made
        if not th.any(_fitns < 0.0):
            break
        # include template with best improvement
        _slctd: int = int(th.argmin(_fitns).item())
        slctd_ms[_slctd] = True
        # update progress bar
        metrics_d: dict[str, float] = {
            "cost": th.mean(th.min(costs[:, slctd_ms], dim=1)[0]).item(),
            "fitness": _fitns[_slctd].item(),
        }
        pbar.set_postfix(metrics_d)
        if log_prefix is not None:
            metrics_d = mylib.utils.add_prefix_to_dict(metrics_d, log_prefix)
        plf.log_dict(metrics_d, _i)
    pbar.close()
    tmpls: th.Tensor = ctmpls[slctd_ms]
    return tmpls, slctd_ms


def make_templates_from_candidates_nearest_neighbors(
    tpcomp: thd.TensorDict,
    ctmpls: th.Tensor,
    n_tmpls: int,
    plf: pl.Fabric,
    log_prefix: Optional[str] = "mk_tmpl_knn",
) -> tuple[th.Tensor, th.Tensor]:
    # (n_data, n_neighs, n_cands)
    knnidxs: th.Tensor = tpcomp["knnidxs"]
    # (n_data, n_cands)
    rwds: th.Tensor = tpcomp["rwds"]
    costs: th.Tensor = -rwds
    costs = th.mean(
        th.gather(costs[:, None, :].expand(-1, len(costs), -1), dim=1, index=knnidxs),
        dim=1,
    )
    # template selected
    # **minimize cost**
    slctd_ms: th.Tensor = th.zeros((len(ctmpls)), dtype=th.bool)
    pbar = tqdm.trange(n_tmpls, desc="make templates", leave=False, dynamic_ncols=True)
    for _i in pbar:
        # start off with best template in the set
        if th.sum(slctd_ms) == 0:
            # (n_cands, )
            _fitns: th.Tensor = th.mean(costs, dim=0)
            _slctd: int = int(th.argmin(_fitns).item())
            slctd_ms[_slctd] = True
            metrics_d: dict[str, float] = {
                "cost": th.mean(th.min(costs[:, slctd_ms], dim=1)[0]).item(),
                "fitness": _fitns[_slctd].item(),
            }
            pbar.set_postfix(metrics_d)
            if log_prefix is not None:
                metrics_d = mylib.utils.add_prefix_to_dict(metrics_d, log_prefix)
            plf.log_dict(metrics_d, _i)
            continue
        # compute currently selected template set cost for each instance
        # (n_data, )
        _slctd_costs = th.min(costs[:, slctd_ms], dim=1)[0]
        # compute potential improvements of each available template for each instance
        # (n_data, n_avail_cands)
        _adj_costs: th.Tensor = costs[:, ~slctd_ms] - _slctd_costs[:, None]
        _adj_costs = th.where(_adj_costs < 0.0, _adj_costs, 0.0)
        # compute average improvements over all instances for each template
        # already selected template is masked out
        # (n_cands, )
        _fitns: th.Tensor = th.zeros((len(ctmpls),))
        _fitns[~slctd_ms] = th.mean(_adj_costs, dim=0)
        # terminate loop early if no improvement is made
        if not th.any(_fitns < 0.0):
            break
        # include template with best improvement
        _slctd: int = int(th.argmin(_fitns).item())
        slctd_ms[_slctd] = True
        # update progress bar
        metrics_d: dict[str, float] = {
            "cost": th.mean(th.min(costs[:, slctd_ms], dim=1)[0]).item(),
            "fitness": _fitns[_slctd].item(),
        }
        pbar.set_postfix(metrics_d)
        if log_prefix is not None:
            metrics_d = mylib.utils.add_prefix_to_dict(metrics_d, log_prefix)
        plf.log_dict(metrics_d, _i)
    pbar.close()
    tmpls: th.Tensor = ctmpls[slctd_ms]
    return tmpls, slctd_ms


@th.no_grad()
def make_templates_vanilla(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls: int,
    n_cands: int,
    min_features: int,
    max_features: Optional[int],
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> th.Tensor:
    n_covs: int = classifier.n_covs
    _i: int = 0
    max_features = n_covs if max_features is None else max_features
    # NOTE init. candidate templates
    ctmpls: th.Tensor = make_template_candidates(
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=n_cands,
        min_features=min_features,
        max_features=max_features,
    )
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(ctmpls)
    tpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
        tmpls=ctmpls,
        data=(
            tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
            if max_tdata is not None and max_tdata < len(tdata)
            else tdata
        ),
        classifier=classifier,
        lmbda=lmbda,
        bsz=bsz,
    )
    tmpls, slctd_ms = make_templates_from_candidates(
        tpcomp=tpcomp,
        ctmpls=ctmpls,
        n_tmpls=n_tmpls,
        plf=plf,
        log_prefix="vanilla_mktmpl",
    )
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(tmpls)
    return tmpls


@th.no_grad()
def make_templates_reduce_features(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_targ: int,
    min_features_targ: int,
    max_features_targ: Optional[int],
    min_features_init: int,
    feature_decrement: int,
    use_feature_importance_sampling: bool,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> th.Tensor:
    n_covs: int = classifier.n_covs
    max_features_targ = n_covs if max_features_targ is None else max_features_targ
    ctmpls: th.Tensor | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    _minfeats_l: list[int] = [
        _minfeats
        for _minfeats in range(
            min_features_init, min_features_targ - 1, -feature_decrement
        )
    ]
    if min_features_targ not in _minfeats_l:
        _minfeats_l.append(min_features_targ)
    _maxfeats: int = max_features_targ
    _i: int = 0
    for _minfeats in tqdm.tqdm(
        _minfeats_l, desc="reduce features", leave=False, dynamic_ncols=True
    ):
        if ctmpls is None or tmpls is None or slctd_ms is None:
            # initialize candidate pool
            ctmpls = make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_targ,
                min_features=_minfeats,
                max_features=_maxfeats,
            )
        else:
            # update candidate pool from existing templates
            _maxfeats = min(max_features_targ, int(th.max(th.sum(tmpls, dim=1)).item()))
            ctmpls = update_template_candidates(
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_targ,
                min_features=_minfeats,
                max_features=_maxfeats,
                use_feature_importance_sampling=use_feature_importance_sampling,
            )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
                if max_tdata is not None and max_tdata < len(tdata)
                else tdata
            ),
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
        )
        tmpls, slctd_ms = make_templates_from_candidates(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"reduce_mktmpl{_i}",
        )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(tmpls)
        _i = _i + 1
    assert tmpls is not None
    return tmpls


@th.no_grad()
def make_templates_fix_rounds(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_init: int,
    min_features: int,
    max_features: Optional[int],
    n_rounds: int,
    use_feature_importance_sampling: bool,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> th.Tensor:
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    ctmpls: th.Tensor | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    for _i in tqdm.trange(
        n_rounds, desc="mktmpl fix rounds", leave=False, dynamic_ncols=True
    ):
        if ctmpls is None or tmpls is None or slctd_ms is None:
            # initialize candidate pool
            ctmpls = make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
            )
        else:
            # update candidate pool from existing templates
            ctmpls = update_template_candidates_fix_rounds(
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                init_fidx=init_fidx,
                min_features=min_features,
                max_features=max_features,
                use_feature_importance_sampling=use_feature_importance_sampling,
            )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
                if max_tdata is not None and max_tdata < len(tdata)
                else tdata
            ),
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
        )
        tmpls, slctd_ms = make_templates_from_candidates(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"fixrounds_mktmpl{_i}",
        )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(tmpls)
    assert tmpls is not None
    return tmpls


@th.no_grad()
def make_templates_fix_rounds_nearest_neighbors(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_init: int,
    min_features: int,
    max_features: Optional[int],
    n_rounds: int,
    use_feature_importance_sampling: bool,
    lmbda: float,
    n_neighs: int,
    bsz: int,
    plf: pl.Fabric,
) -> th.Tensor:
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    ctmpls: th.Tensor | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    for _i in tqdm.trange(
        n_rounds, desc="mktmpl fix rounds", leave=False, dynamic_ncols=True
    ):
        if ctmpls is None or tmpls is None or slctd_ms is None:
            # initialize candidate pool
            ctmpls = make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
            )
        else:
            # update candidate pool from existing templates
            ctmpls = update_template_candidates_fix_rounds(
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                init_fidx=init_fidx,
                min_features=min_features,
                max_features=max_features,
                use_feature_importance_sampling=use_feature_importance_sampling,
            )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(ctmpls)
        _tdata = (
            tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
            if max_tdata is not None and max_tdata < len(tdata)
            else tdata
        )
        tpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
            ctmpls,
            data=_tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
        )
        tpcomp["knnidxs"] = th.empty(
            (len(_tdata), n_neighs, len(ctmpls)), dtype=th.long
        )
        for _j, m in tqdm.tqdm(
            enumerate(ctmpls.to(dtype=th.bool)),
            desc="tpcomp knn",
            total=len(ctmpls),
            leave=False,
            dynamic_ncols=True,
        ):
            _dists: th.Tensor = th.as_tensor(
                sp_spatial.distance.squareform(
                    th.pdist(_tdata["xs"][:, m]).numpy(force=True)
                ),
                dtype=th.float32,
            )
            tpcomp["knnidxs"][:, :, _j] = th.argsort(_dists, dim=1, descending=False)[
                :, 1 : n_neighs + 1
            ]
        tmpls, slctd_ms = make_templates_from_candidates_nearest_neighbors(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"fixrounds_mktmpl{_i}_knn",
        )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(tmpls)
    assert tmpls is not None
    return tmpls


@th.no_grad()
def _knn(
    xs: th.Tensor, txs: th.Tensor, n_neighs: int, p: float = 2
) -> tuple[th.Tensor, th.Tensor]:
    """knn

    Args:
        xs (th.Tensor): (n, n_covs) inputs
        txs (th.Tensor): (n_tdata, n_covs) training input covariates
        n_neighs (int): number of neighbors
        p (float, optional): p-norm distance. Defaults to 2.

    Returns::
        th.Tensor: (n, n_neighs) distance to nearest neighbors neighbors
        th.Tensor: (n, n_neighs) indices to nearest neighbors in training data
    """
    # (n, n_tdata)
    ds: th.Tensor = th.cdist(xs[None, :, :], txs[None, :, :], p=p)[0]
    # (n, n_tdata)
    ds, knnidxs = th.sort(ds, dim=1, descending=False)
    ds = ds[:, :n_neighs]
    knnidxs = knnidxs[:, :n_neighs]
    return ds, knnidxs


def knn_cost_est(
    inps: th.Tensor,
    lmbda: float,
    txs: th.Tensor,
    tcels: th.Tensor,
    tmpls: th.Tensor,
    n_neighs: int,
    p: float = 2,
    is_train: bool = False,
) -> th.Tensor:
    """use knn strategy to compute cost

    Args:
        inps (th.Tensor): (n, n_covs * 2)
        lmbda (float): feature penalty
        txs (th.Tensor): (n, n_covs)
        tcels (th.Tensor): (n, n_tmpls)
        tmpls (th.Tensor): (n_tmpls, n_covs)
        n_neighs (int): number of neighbors
        p (float, optional): p-norm distance. Defaults to 2.
        is_train (bool, optional): whether the given `inps` is from training set. Defaults to `False`.

    Returns::
        th.Tensor: (n, n_tmpls) costs of using each template
    """
    # th.Tensor: (n, n_tmpls) costs of using each template
    n_covs: int = txs.shape[1]
    device: th.device = inps.device
    txs = txs.to(device=device)
    tcels = tcels.to(device=device)
    costs_l: list[th.Tensor] = list()
    n_neighs = n_neighs + 1 if is_train else n_neighs
    for _inp in inps:
        # (n_covs, )
        _inp: th.Tensor
        _fm: th.Tensor = _inp[n_covs:]
        _fc: tuple[int, ...] = tuple(th.argwhere(_fm == 1).flatten().tolist())
        # (n_tmpls, n_covs)
        _fm_avail: th.Tensor = th.maximum(tmpls - _fm[None, :], th.as_tensor(0.0))
        # (n_neighs, )
        _knnidxs: th.Tensor = _knn(
            xs=_inp[None, _fc].to(device=device),
            txs=txs[:, _fc].to(device=device),
            n_neighs=n_neighs,
            p=p,
        )[1][0].to(device="cpu")
        _knnidxs = _knnidxs[1:] if is_train else _knnidxs
        # (n_tmpls, )
        _cels: th.Tensor = th.mean(tcels[_knnidxs], dim=0)
        _costs: th.Tensor = _cels + lmbda * th.sum(_fm_avail, dim=1)
        costs_l.append(_costs)
    costs: th.Tensor = th.stack(costs_l, dim=0).to(device=device)
    return costs


def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        _costs: th.Tensor = cost_est(_inps.to(device=plf.device)).to(device="cpu")
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0))
        if th.sum(_fm_avail) == 0:
            fcomb = tuple(th.argwhere(tmpls[_tmpl_idx] == 1).flatten().tolist())
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    assert fcomb is not None
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fcomb] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


def run_one_episode_all_obsd(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        _costs: th.Tensor = cost_est(_inps.to(device=plf.device)).to(device="cpu")
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0))
        if th.sum(_fm_avail) == 0:
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    fcomb = tuple(sorted(fobsd_l))
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fobsd_l] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


def evaluate(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
) -> dict[str, float]:
    snfobsd_l: list[int] = list()
    snfcomb_l: list[int] = list()
    metrics_func.reset()
    for _data in data:
        _pyhat, _fobsd_l, _fcomb = run_one_episode_all_obsd(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        snfobsd_l.append(len(_fobsd_l))
        snfcomb_l.append(len(_fcomb))
        metrics_func.update(
            _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
        )
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    metrics_d.update(
        {
            "init_fidx": init_fidx,
            "feature observed": th.mean(
                th.as_tensor(snfobsd_l, dtype=th.float32)
            ).item(),
            "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
        }
    )
    return metrics_d
