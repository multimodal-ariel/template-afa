# %%
from __future__ import annotations

import itertools as itrtls
import math
import os
from typing import Any, Optional, TypedDict

import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import numpy as np
import pandas as pd
import tensordict as thd
import torch as th
import torch.distributions.utils
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst


# %%
class SubsetFeatureNaiveBayes(mymodels.classifiers.SubsetFeatureClassifier[None]):
    std: float

    def __init__(self, std: float, xs_train: np.ndarray, ys_train: np.ndarray):
        super().__init__(n_experts_per_act=1, xs_train=xs_train, ys_train=ys_train)
        self.std = std

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        pyhats: th.Tensor = self._aaco_forward_impl(
            ctxs.to(device="cpu"), acts.to(device="cpu")
        ).to(device=self.device)
        return pyhats

    def _aaco_forward_impl(self, x: th.Tensor, mask: th.Tensor):
        from scipy.stats import norm

        y_classes = list(range(self.n_labels))

        output_probs = th.zeros((len(x), self.n_labels))

        for y_val in y_classes:

            ## PDF values for each feature in x conditioned on the given label y_val

            # Default to PDF for U[0,1)
            p_x_y = th.where((x >= 0) & (x < 1), th.ones(x.shape), th.zeros(x.shape))

            # Use normal distribution PDFs for appropriate features given y_val
            p_x_y[:, y_val : y_val + 3] = th.transpose(
                th.Tensor(
                    np.array(
                        [
                            norm.pdf(x[:, y_val], y_val % 2, self.std),
                            norm.pdf(x[:, y_val + 1], (y_val // 2) % 2, self.std),
                            norm.pdf(x[:, y_val + 2], (y_val // 4) % 2, self.std),
                        ]
                    )
                ),
                0,
                1,
            )

            # Compute joint probability over masked features
            p_xo_y = th.prod(
                th.where(th.gt(mask, 0), p_x_y, th.tensor(1).float()), dim=1
            )

            p_y = 1 / self.n_labels

            output_probs[:, y_val] = p_xo_y * p_y

        return th.divide(
            output_probs,
            th.squeeze(th.dstack([th.sum(output_probs, dim=1)] * self.n_labels)),
        )

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None


class SubsetFeatureConcatXGBClassifier(
    mymodels.classifiers.SubsetFeatureClassifier[None]
):
    xgb_kwargs: dict[str, Any]
    fraction_training_data_per_split: float
    n_splits: int
    n_tmpl_per_instance: int
    rseed: Optional[int]

    _models: list[xgbst.XGBClassifier]

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        xgb_kwargs: dict[str, Any],
        fraction_training_data_per_split: float,
        n_splits: int,
        n_tmpl_per_instance: int,
        rseed: Optional[int] = None,
    ):
        super().__init__(n_experts_per_act=1, xs_train=xs_train, ys_train=ys_train)
        self.xgb_kwargs = xgb_kwargs
        self.fraction_training_data_per_split = fraction_training_data_per_split
        self.n_splits = n_splits
        self.n_tmpl_per_instance = n_tmpl_per_instance
        self.rseed = rseed
        self._models = [xgbst.XGBClassifier(**self.xgb_kwargs) for _ in range(n_splits)]

    def fit_(self, tmpls: th.Tensor):
        txs: th.Tensor = th.as_tensor(self.xs_train, dtype=th.float32)
        tys: th.Tensor = th.as_tensor(self.ys_train)
        pbar = tqdm.tqdm(
            self._models, desc="model rsplit", dynamic_ncols=True, leave=True
        )
        for _m in pbar:
            _n_data: int = math.ceil(len(txs) * self.fraction_training_data_per_split)
            _idxs: th.Tensor = th.randint(0, len(txs), size=(_n_data,), dtype=th.long)
            # (_n_data, n_tmpl_per_instance, n_covs)
            _xs: th.Tensor = txs[_idxs, None, :].expand(
                -1, self.n_tmpl_per_instance, -1
            )
            _fms: th.Tensor = th.stack(
                [
                    tmpls[
                        th.multinomial(
                            th.arange(0, len(tmpls), dtype=th.float32),
                            self.n_tmpl_per_instance,
                        )
                    ]
                    for _ in range(len(_xs))
                ]
            )
            # (_n_data, n_tmple_per_instance, 2 * n_covs)
            _minps: th.Tensor = th.cat((_xs * _fms, _fms), dim=2)
            # (_n_data, n_tmpl_per_instance)
            _ys: th.Tensor = tys[_idxs, None].expand(-1, self.n_tmpl_per_instance)
            _m.fit(_minps.flatten(0, 1).numpy(), _ys.flatten(0, 1).numpy())
        pbar.close()

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        device: th.device = ctxs.device
        ctxs = ctxs.to(device="cpu")
        acts = acts.to(device="cpu")
        # (n, n_covs * 2)
        minps: th.Tensor = th.cat((ctxs * acts, acts), dim=1)
        # (n, n_splits, n_labels)
        pyhats: th.Tensor = th.stack(
            [
                th.as_tensor(
                    _m.predict_proba(minps.numpy(force=True)), dtype=th.float32
                )
                for _m in self._models
            ],
            dim=1,
        )
        # (n, n_labels)
        pyhats = th.mean(pyhats, dim=1).to(device=device)
        return pyhats

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None


# %%
class SoftmaxSelector(th.nn.Module):
    n_covs: int
    n_templates: int
    nnet: th.nn.Module

    def __init__(self, n_covs: int, n_tmpls: int, nnet: th.nn.Module) -> None:
        super().__init__()
        self.n_covs = n_covs
        self.n_templates = n_tmpls
        self.nnet = nnet

    def forward(self, sinps: th.Tensor, to_probs: bool) -> th.Tensor:
        outs: th.Tensor = self.nnet(sinps)
        if to_probs:
            outs = th.nn.functional.softmax(outs)
        return outs


# %%
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


@th.no_grad()
def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    selector: SoftmaxSelector,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    selector.eval().to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    # repeat feature acquisition until all features in template has been acquired.
    for _ in itrtls.count():
        # make feature bit mask
        _m: th.Tensor = th.zeros_like(x)
        _m[fobsd_l] = 1
        # forward prop. selector
        _sinps: th.Tensor = th.cat((x, _m))[None, :].to(device=plf.device)
        _souts: th.Tensor = selector(_sinps, to_probs=False)
        # choose a template
        _fcomb_idx: int = int(th.argmax(_souts[0]).item())
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


# %%
def make_template_candidates(
    n_covs: int, init_fidx: int, n_cands: int, min_features: int, max_features: int
) -> th.Tensor:
    bincnt_fcs_l: list[int] = [
        # in order to accomondate for init_fidx,
        # both n_covs and i is one less than desired n_feats
        math.comb(n_covs - 1, i)
        for i in range(
            min_features - 1, n_covs if max_features is None else max_features
        )
    ]
    assert n_cands <= sum(
        bincnt_fcs_l
    ), "n_cands is larger than all possible feature combinations"
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


# NOTE use this one if one day pytorch fix expand.flatten operation memory usage
# def precomp_rwds_for_ctmpls(
#     ctmpls: th.Tensor,
#     tdata: thd.TensorDict,
#     classifier: mymodels.classifiers.SubsetFeatureClassifier,
#     lmbda: float,
#     bsz: int,
# ) -> thd.TensorDict:
#     txs: th.Tensor = tdata["xs"]
#     tys: th.Tensor = tdata["ys"]
#     n_cands: int = len(ctmpls)
#     n_labels: int = len(th.unique(tys))
#     # (n_data * n_cands, n_covs)
#     ctxs: th.Tensor = txs[:, None, :].expand(-1, n_cands, -1).flatten(0, 1)
#     acts: th.Tensor = ctmpls[None, :, :].expand(len(txs), -1, -1).flatten(0, 1)
#     tys_: th.Tensor = tys[:, None].expand(-1, n_cands).flatten()
#     # (n_data * n_cands, n_labels)
#     pyhats: th.Tensor = th.empty((len(txs) * n_cands, n_labels), dtype=th.float32)
#     # (n_data * n_cands)
#     cels: th.Tensor = th.empty((len(txs) * n_cands,), dtype=th.float32)
#     rwds: th.Tensor = th.empty_like(cels)
#     pbar = tqdm.tqdm(
#         th.split(th.arange(0, len(ctxs), dtype=th.long), bsz),
#         desc="precomp candidates",
#         leave=False,
#         dynamic_ncols=True,
#     )
#     for _btidxs in pbar:
#         _bpyhats: th.Tensor = classifier.predict_proba(ctxs[_btidxs], acts[_btidxs])
#         _blyhats: th.Tensor = torch.distributions.utils.probs_to_logits(_bpyhats)
#         _bcels: th.Tensor = th.nn.functional.cross_entropy(
#             _blyhats, tys_[_btidxs], reduction="none"
#         )
#         pyhats[_btidxs] = _bpyhats
#         cels[_btidxs] = _bcels
#         rwds[_btidxs] = -_bcels - lmbda * th.sum(acts[_btidxs], dim=1)
#     pbar.close()
#     # (n_data, n_cands, n_labels)
#     pyhats = pyhats.unflatten(0, (len(txs), n_cands))
#     # (n_data, n_cands)
#     cels = cels.unflatten(0, (len(txs), n_cands))
#     rwds = rwds.unflatten(0, (len(txs), n_cands))
#     # turn into tensordict
#     tpcomp = thd.TensorDict(
#         {"pyhats": pyhats, "cels": cels, "rwds": rwds}
#     ).auto_batch_size_(1)
#     return tpcomp


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


def make_templates(
    tpcomp: thd.TensorDict, ctmpls: th.Tensor, n_tmpls: int
) -> tuple[th.Tensor, th.Tensor]:
    # (n_data, n_cands)
    rwds: th.Tensor = tpcomp["rwds"]
    costs: th.Tensor = -rwds
    # template selected
    # **minimize cost**
    slctd_ms: th.Tensor = th.zeros((len(ctmpls)), dtype=th.bool)
    pbar = tqdm.trange(n_tmpls, desc="make templates", leave=True, dynamic_ncols=True)
    for _i in pbar:
        # start off with best template in the set
        if th.sum(slctd_ms) == 0:
            # (n_cands, )
            _fitns: th.Tensor = th.mean(costs, dim=0)
            _slctd: int = int(th.argmin(_fitns).item())
            slctd_ms[_slctd] = True
            pbar.set_postfix(
                {
                    "cost": th.mean(th.min(costs[:, slctd_ms], dim=1)[0]).item(),
                    "fitness": _fitns[_slctd].item(),
                }
            )
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
        pbar.set_postfix(
            {
                "cost": th.mean(th.min(costs[:, slctd_ms], dim=1)[0]).item(),
                "fitness": _fitns[_slctd].item(),
            }
        )
    pbar.close()
    tmpls: th.Tensor = ctmpls[slctd_ms]
    return tmpls, slctd_ms


@th.no_grad()
def compile_selector_dataset(
    tdata: thd.TensorDict,
    tpcomp: thd.TensorDict,
    slctd_ms: th.Tensor,
) -> thd.TensorDict:
    # (n_data, n_covs)
    xs: th.Tensor = tdata["xs"]
    # (n_data, )
    ys: th.Tensor = tdata["ys"]
    # (n_data, n_tmpls, n_labels)
    pyhats: th.Tensor = tpcomp["pyhats"][:, slctd_ms, :]
    # (n_data, n_tmpls)
    cels: th.Tensor = tpcomp["cels"][:, slctd_ms]
    rwds: th.Tensor = tpcomp["rwds"][:, slctd_ms]
    # (n_data, )
    slbls: th.Tensor = th.argmax(rwds, dim=1)
    # # (n_data, n_tmpls)
    # slbls: th.Tensor = th.softmax(rwds / tau_rwd, dim=1)
    # bundle tensors into tensordict
    stdata = thd.TensorDict(
        {
            "xs": xs,
            "ys": ys,
            "pyhats": pyhats,
            "cels": cels,
            "rwds": rwds,
            "slbls": slbls,
        }
    ).auto_batch_size_(1)
    return stdata


class _TrainState(TypedDict):
    selector: SoftmaxSelector
    opt: th.optim.Optimizer
    n_trial_itr: int
    n_fit_itr: int
    opt_step: int


@th.no_grad()
def _make_fit_bsinps(
    bstdata: thd.TensorDict, init_fidx: int, tmpls: th.Tensor
) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bfms: th.Tensor = tmpls[btmplidxs].to(device=bxs.device)
    n_covs: int = bxs.shape[1]
    # randomly drop features
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs))
    bnms[:, init_fidx] = 1
    bnms = th.cat((bnms, bnms), dim=1).to(device=bxs.device)
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs, bfms), dim=1)
    bsinps = bsinps * bnms
    return bsinps


def _fit_iter(
    tstate: _TrainState,
    tloader: th_data.DataLoader,
    init_fidx: int,
    tmpls: th.Tensor,
    pbar: tqdm.tqdm,
    plf: pl.Fabric,
) -> dict[str, float]:
    selector: SoftmaxSelector = tstate["selector"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    slosses_l: list[th.Tensor] = list()
    for bstdata in tloader:
        bstdata: thd.TensorDict
        bstdata = bstdata.to(device=plf.device)
        # (bsz, 2 * n_covs)
        bsinps: th.Tensor = _make_fit_bsinps(
            bstdata=bstdata, init_fidx=init_fidx, tmpls=tmpls
        )
        # (bsz, n_tmpls)
        bsouts: th.Tensor = selector(bsinps, to_probs=False)
        # compute selector loss
        bslosses: th.Tensor = th.nn.functional.cross_entropy(
            bsouts, bstdata["slbls"], reduction="none"
        )
        bsloss: th.Tensor = th.mean(bslosses)
        # update selector parameter
        opt.zero_grad()
        bsloss.backward()
        opt.step()
        # track metrics
        slosses_l.append(bslosses.detach().to(device="cpu"))
        bmetrics_d: dict[str, float] = {
            "bsloss": bsloss.item(),
        }
        pbar.set_postfix(bmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(bmetrics_d, "train"), step=tstate["opt_step"]
        )
        tstate["opt_step"] = tstate["opt_step"] + 1
    sloss_avg: th.Tensor = th.mean(th.cat(slosses_l, dim=0))
    metrics_d: dict[str, float] = {"sloss_avg": sloss_avg.item()}
    return metrics_d


def fit(
    tstate: _TrainState,
    stdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
):
    tloader = th_data.DataLoader(
        stdata,  # type: ignore
        batch_size=bsz,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _ in pbar:
        metrics_d: dict[str, float] = _fit_iter(
            tstate=tstate,
            tloader=tloader,
            init_fidx=init_fidx,
            tmpls=tmpls,
            pbar=pbar,
            plf=plf,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "train"), step=tstate["n_fit_itr"]
        )
        tstate["n_fit_itr"] = tstate["n_fit_itr"] + 1
    pbar.close()


# %%
# data_name: str = "cube_20_0.3"
# tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# n_covs: int = tdata["xs"].shape[1]
# n_labels: int = len(th.unique(tdata["ys"]))

# # %%
# classifier = SubsetFeatureNaiveBayes(
#     0.3, xs_train=tdata["xs"].numpy(), ys_train=tdata["ys"].numpy()
# )
# metrics_func = thm.MetricCollection(
#     {
#         "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
#         "precision": thm.Precision(task="multiclass", num_classes=n_labels),
#         "recall": thm.Recall(task="multiclass", num_classes=n_labels),
#         "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
#         "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
#     }
# )

# #  %%
# init_fidx: int = 6
# n_tmpls: int = 64
# n_cands: int = 5_000
# lmbda: float = 0.0
# max_features: int = 5
# # tau_rwd: float = 0.01
# bsz: int = 1024

# %%
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))

# %%
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
tdata = tdata[:6000]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
classifier = SubsetFeatureConcatXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgb_kwargs={"n_estimators": 40},
    fraction_training_data_per_split=1.0,
    n_splits=64,
    n_tmpl_per_instance=4,
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

#  %%
init_fidx: int = 31
n_tmpls: int = 128
n_cands: int = 10_000
lmbda: float = 0.075
max_features: int = n_covs
# tau_rwd: float = 0.01
bsz: int = 1024

# %%
ctmpls: th.Tensor = make_template_candidates(
    n_covs=n_covs,
    init_fidx=init_fidx,
    n_cands=n_cands,
    min_features=1,
    max_features=max_features,
)

# %%
classifier.fit_(tmpls=ctmpls)

# %%
tpcomp = precomp_rwds_for_tmpls(
    ctmpls, data=tdata, classifier=classifier, lmbda=lmbda, bsz=bsz
)

# %%
tmpls, slctd_ms = make_templates(tpcomp=tpcomp, ctmpls=ctmpls, n_tmpls=n_tmpls)
allfcombs_l: list[tuple[int, ...]] = [
    tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls
]
n_tmpls = len(tmpls)

# %%
print("greedy+random")
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = run_one_random_episode(
        x=_data["xs"],
        classifier=classifier,
        init_fidx=init_fidx,
        allfcombs_l=allfcombs_l,
    )
    snfobsd_l.append(len(_fobsd_l))
    snfcomb_l.append(len(_fcomb))
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "feature observed": th.mean(th.as_tensor(snfobsd_l), dtype=th.float32).item(),
        "feature used": th.mean(th.as_tensor(snfcomb_l), dtype=th.float32).item(),
    }
)
print(pd.Series(metrics_d))

# %%
vpcomp = precomp_rwds_for_tmpls(
    tmpls=tmpls, data=vdata, classifier=classifier, lmbda=lmbda, bsz=bsz
)

# %%
print("greedy+oracle")
acts, pyhats, ys, _ = eval_with_oracle_from_precomp(
    data=vdata, pcomp=vpcomp, tmpls=tmpls
)
metrics_func.reset()
metrics_func.update(pyhats[:, :, None], ys[:, None])
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "feature observed": th.mean(th.sum(acts, dim=1).to(dtype=th.float32)).item(),
        "feature used": th.mean(th.sum(acts, dim=1).to(dtype=th.float32)).item(),
    }
)
print(pd.Series(metrics_d))

# %%
# stdata = compile_selector_dataset(tdata=tcube, tpcomp=tpcomp, slctd_ms=slctd_ms)

# # %%
# # configure logger and ckpt path
# output_dir: str = os.path.join("outputs", "run", data_name, "slt_fxgreedy")
# os.makedirs(output_dir, exist_ok=True)
# tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")

# # %%
# plf = pl.Fabric(loggers=[tfb_logger])
# # plf = pl.Fabric(loggers=[tfb_logger], accelerator="cpu")

# # %%
# nnet = mymodels.nn.make_fcn(
#     in_features=2 * tcube["xs"].shape[1],
#     out_features=n_tmpls,
#     layer_specs=[
#         (tcube["xs"].shape[1], None, None, None),
#         (tcube["xs"].shape[1], None, None, None),
#     ],
# )
# selector = SoftmaxSelector(
#     n_covs=n_covs,
#     n_tmpls=n_tmpls,
#     nnet=nnet,
# )
# opt = th.optim.Adam(selector.parameters())

# # %%
# tstate = _TrainState(selector=selector, opt=opt, n_trial_itr=0, n_fit_itr=0, opt_step=0)

# # %%
# fit(
#     tstate=tstate,
#     stdata=stdata,
#     init_fidx=init_fidx,
#     tmpls=tmpls,
#     n_iter=5000,
#     bsz=4096,
#     plf=plf,
# )

# # %%
# print("greedy+selector")
# metrics_func.reset()
# snfobsd_l: list[int] = list()
# snfcomb_l: list[int] = list()
# for _data in vcube:
#     _pyhat, _fobsd_l, _fcomb = run_one_episode(
#         x=_data["xs"],
#         classifier=classifier,
#         selector=selector,
#         init_fidx=init_fidx,
#         allfcombs_l=allfcombs_l,
#         plf=plf,
#     )
#     snfobsd_l.append(len(_fobsd_l))
#     snfcomb_l.append(len(_fcomb))
#     metrics_func.update(
#         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
#     )
# metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# metrics_func.reset()
# metrics_d.update(
#     {
#         "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
#         "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
#     }
# )
# print(pd.Series(metrics_d))

# %%
