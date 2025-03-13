from __future__ import annotations

import math
from typing import Optional

import torch as th


# NOTE identify initial feature
def make_feature_masks(
    n_covs: int,
    n_masks: int,
    min_features: int,
    max_features: Optional[int],
) -> th.Tensor:
    """make random subset feature masks

    Args:
        n_covs (int): number of covariates
        n_masks (int): number of masks
        min_features (int): minimum number of features enabled in each mask
        max_features (Optional[int]): maximum number of features enabled in each masks; `max_features = n_covs` if `None`

    Returns:
        th.Tensor: (n_masks_actual, n_covs) feature masks
    """
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


def make_template_candidates(
    n_covs: int,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
) -> th.Tensor:
    """make random candidate set of subset feature masks

    Args:
        n_covs (int): number of covariates
        init_fidx (int): initial feature index
        n_cands_targ (int): number of candidate feature masks
        min_features (int): minimum number of features enabled in each mask
        max_features (Optional[int]): maximum number of features enabled in each masks; `max_features = n_covs` if `None`

    Returns:
        th.Tensor: (n_cands_actual, n_covs) feature masks
    """
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
