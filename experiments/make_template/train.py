from __future__ import annotations

import itertools as itrtls
import math
import os
from typing import Any, Optional, TypedDict

from attr import dataclass
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import numpy as np
import tensordict as thd
import torch as th
import torch.distributions.utils
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@dataclass
class MainConf:
    data: Any
    classifier: Any
    init_fidx: int
    n_tmpls_targ: int
    n_cands_targ: int
    min_features_targ: int
    max_features_targ: Optional[int]
    min_features_init: int
    feature_decrement: int
    lmbda: float
    bsz: int


def run_one_random_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    # repeat feature acquisition until all features in template has been acquired.
    for _ in itrtls.count():
        # randomly choose an initial template
        _fcomb_idx: int = int(th.randint(0, len(allfcombs_l), size=(1,)).item())
        _tmpl_fcomb: tuple[int, ...] = allfcombs_l[_fcomb_idx]
        # ident. unacquired features
        _tmp_fcomb: list[int] = [fidx for fidx in _tmpl_fcomb if fidx not in fobsd_l]
        if len(_tmp_fcomb) == 0:
            fcomb = _tmpl_fcomb
            break
        # randomly choose a feature to acquire
        _tmp_fcomb_idx = int(th.randint(0, len(_tmp_fcomb), size=(1,)).item())
        # add acquired feature to fcomb
        fobsd_l.append(_tmp_fcomb[_tmp_fcomb_idx])
        # terminate acq. if all features in template has been satisfied
        if all([fidx in fobsd_l for fidx in _tmpl_fcomb]):
            fcomb = _tmpl_fcomb
            break
    assert fcomb is not None
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fcomb] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :], acts)
    return pyhats[0], fobsd_l, fcomb


def eval_with_oracle_from_precomp(
    data: thd.TensorDict, pcomp: thd.TensorDict, tmpls: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    # (n_data, n_tmpl)
    rwds: th.Tensor = pcomp["rwds"]
    # (n_data, ) (n_data, )
    rwds, aidxs = th.max(rwds, dim=1)
    # (n_data, n_covs)
    acts: th.Tensor = tmpls[aidxs]
    # (n_data, n_labels)
    pyhats: th.Tensor = th.gather(
        pcomp["pyhats"], dim=1, index=aidxs[:, None, None].expand_as(pcomp["pyhats"])
    )[:, 0, :]
    return acts, pyhats, data["ys"], rwds


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
        math.comb(n_covs - 1, i)
        for i in range(
            min_features - 1, n_covs if max_features is None else max_features
        )
    ]
    n_cands: int = min(n_cands_targ, sum(bincnt_fcs_l))
    bincnt_fcs: th.Tensor = th.as_tensor(bincnt_fcs_l)
    ps: th.Tensor = bincnt_fcs / th.sum(bincnt_fcs)
    ps = ps.to(dtype=th.float64)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_cands, replacement=True), minlength=len(bincnt_fcs)
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(
            _curr_bincnts >= bincnt_fcs, 0, bincnt_fcs - _curr_bincnts
        )
        _tmp_ps = _tmp_ps / th.sum(_tmp_ps)
        _tmp_ps = _tmp_ps.to(dtype=th.float64)
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
        _nfeats_prv: int = int(th.sum(_tmpl_prv).item())
        # compute number of features that can be mutated
        _nfeats_mutavail = _nfeats_prv - min_features
        # make copy of selected template
        _tmpl: th.Tensor = _tmpl_prv.clone()
        # number of features to mutate
        _nfeats_mut: int = int(th.randint(1, _nfeats_mutavail + 1, ()).item())
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
) -> list[set[tuple[int, ...]]]:
    # fill the rest of the demanded with randomly generated ones
    # group fcs_set by len(fcomb)
    max_features = n_covs if max_features is None else max_features
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [
        set() for _ in range(min_features - 1, max_features)
    ]
    [fcs_sets_by_bins[len(_fc) - min_features].add(_fc) for _fc in fcs_set]
    if len(fcs_set) >= n_cands_targ:
        return fcs_sets_by_bins
    # compute maximum feature combinations allowed in each bin
    bincnt_fcs: th.Tensor = th.as_tensor(
        [
            # in order to accomondate for init_fidx,
            # both n_covs and i is one less than desired n_feats
            math.comb(n_covs - 1, i)
            for i in range(
                min_features - 1, n_covs if max_features is None else max_features
            )
        ]
    )
    # subtract existing fcs from the bins
    bincnt_fcs = bincnt_fcs - th.as_tensor(
        list(map(len, fcs_sets_by_bins)), dtype=th.long
    )
    # sample number of fcs to add to existing feature combinations
    ps: th.Tensor = bincnt_fcs / th.sum(bincnt_fcs)
    ps = ps.to(dtype=th.float64)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_cands_targ - len(fcs_set), replacement=True),
        minlength=len(bincnt_fcs),
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(
            _curr_bincnts >= bincnt_fcs, 0, bincnt_fcs - _curr_bincnts
        )
        _tmp_ps = _tmp_ps / th.sum(_tmp_ps)
        _tmp_ps = _tmp_ps.to(dtype=th.float64)
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
    return fcs_sets_by_bins


def update_template_candidates(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
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
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = min(n_cands_targ, len(fcs_l))
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
    cels: th.Tensor = th.empty(
        (
            len(txs),
            n_cands,
        ),
        dtype=th.float32,
    )
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


def _eval(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    tmpls: th.Tensor,
    lmbda: float,
    bsz: int,
    metrics_func: thm.MetricCollection,
) -> dict[str, float]:
    vpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
        tmpls=tmpls, data=data, classifier=classifier, lmbda=lmbda, bsz=bsz
    )
    acts, pyhats, ys, rwds = eval_with_oracle_from_precomp(
        data=data, pcomp=vpcomp, tmpls=tmpls
    )
    metrics_func.reset()
    metrics_func.update(pyhats[:, :, None], ys[:, None])
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    metrics_d.update(
        {
            "rwd": th.mean(rwds).item(),
            "feature observed": th.mean(
                th.sum(acts, dim=1).to(dtype=th.float32)
            ).item(),
            "feature used": th.mean(th.sum(acts, dim=1).to(dtype=th.float32)).item(),
        }
    )
    return metrics_d


def make_templates_vanilla(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls: int,
    n_cands: int,
    min_features: int,
    max_features: Optional[int],
    lmbda: float,
    bsz: int,
    vdata: Optional[thd.TensorDict],
    metrics_func: thm.MetricCollection,
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
        tmpls=ctmpls, data=tdata, classifier=classifier, lmbda=lmbda, bsz=bsz
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
    if vdata is not None:
        metrics_d: dict[str, float] = _eval(
            data=vdata,
            classifier=classifier,
            tmpls=tmpls,
            lmbda=lmbda,
            bsz=bsz,
            metrics_func=metrics_func,
        )
        metrics_d.update(
            {
                "minfeats": min_features,
                "maxfeats": max_features,
            }
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "train_vanilla"), _i)
    return tmpls


def make_templates_reduce_features(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_targ: int,
    min_features_targ: int,
    max_features_targ: Optional[int],
    min_features_init: int,
    feature_decrement: int,
    lmbda: float,
    bsz: int,
    vdata: Optional[thd.TensorDict],
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
) -> th.Tensor:
    n_covs: int = classifier.n_covs
    _i: int = 0
    max_features_targ = n_covs if max_features_targ is None else max_features_targ
    # NOTE init. candidate templates
    ctmpls: th.Tensor = make_template_candidates(
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=n_cands_targ,
        min_features=min_features_init,
        max_features=max_features_targ,
    )
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(ctmpls)
    tpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
        tmpls=ctmpls, data=tdata, classifier=classifier, lmbda=lmbda, bsz=bsz
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
    if vdata is not None:
        metrics_d: dict[str, float] = _eval(
            data=vdata,
            classifier=classifier,
            tmpls=tmpls,
            lmbda=lmbda,
            bsz=bsz,
            metrics_func=metrics_func,
        )
        metrics_d.update(
            {
                "minfeats": min_features_init,
                "maxfeats": max_features_targ,
            }
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "train_reduce"), _i)
    # NOTE start decreasing features
    for _minfeats in tqdm.trange(
        min_features_init - feature_decrement,
        min_features_targ - 1,
        -feature_decrement,
        desc="reduce features",
        leave=False,
        dynamic_ncols=True,
    ):
        _i = _i + 1
        _maxfeats: int = min(
            max_features_targ, int(th.max(th.sum(tmpls, dim=1)).item())
        )
        ctmpls = update_template_candidates(
            ctmpls=ctmpls,
            slctd_ms=slctd_ms,
            init_fidx=init_fidx,
            n_cands_targ=n_cands_targ,
            min_features=_minfeats,
            max_features=_maxfeats,
        )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(ctmpls)
        tpcomp = precomp_rwds_for_tmpls(
            ctmpls, data=tdata, classifier=classifier, lmbda=lmbda, bsz=bsz
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
        if vdata is not None:
            metrics_d: dict[str, float] = _eval(
                data=vdata,
                classifier=classifier,
                tmpls=tmpls,
                lmbda=lmbda,
                bsz=bsz,
                metrics_func=metrics_func,
            )
            metrics_d.update(
                {
                    "minfeats": _minfeats,
                    "maxfeats": _maxfeats,
                }
            )
            plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "train_reduce"), _i)
    return tmpls


@hd.main(version_base=None)
def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # make classifier
    classifier: mymodels.classifiers.SubsetFeatureClassifier
    classifier = hd.utils.instantiate(
        cfg.classifier, xs_train=tdata["xs"].numpy(), ys_train=tdata["ys"].numpy()
    )
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    # configure logger and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")
    plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="cpu")
    # NOTE vanilla
    tmpls = make_templates_vanilla(
        tdata=tdata,
        classifier=classifier,
        init_fidx=cfg.init_fidx,
        n_tmpls=cfg.n_tmpls_targ,
        n_cands=cfg.n_cands_targ,
        min_features=cfg.min_features_targ,
        max_features=cfg.max_features_targ,
        lmbda=cfg.lmbda,
        bsz=cfg.bsz,
        vdata=vdata,
        metrics_func=metrics_func,
        plf=plf,
    )
    tmpls = make_templates_reduce_features(
        tdata=tdata,
        classifier=classifier,
        init_fidx=cfg.init_fidx,
        n_tmpls_targ=cfg.n_tmpls_targ,
        n_cands_targ=cfg.n_cands_targ,
        min_features_targ=cfg.min_features_targ,
        max_features_targ=cfg.max_features_targ,
        min_features_init=cfg.min_features_init,
        feature_decrement=cfg.feature_decrement,
        lmbda=cfg.lmbda,
        bsz=cfg.bsz,
        vdata=vdata,
        metrics_func=metrics_func,
        plf=plf,
    )
    # logger flush record and close
    tfb_logger.finalize("success")
    csv_logger.finalize("success")


if __name__ == "__main__":
    main()
