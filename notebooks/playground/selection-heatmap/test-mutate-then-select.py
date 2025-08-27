# %%
from __future__ import annotations

import itertools as itrtls
import math
import os
from typing import Any, Literal, Optional, TypedDict

import _classifiers
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mylib
import mymodels.classifiers
import pandas as pd
import sklearn.feature_selection as skl_fsel
import sklearn.preprocessing as skl_preproc
import tafalib.functional
import tafalib.makers.templates
import tafalib.utils
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = mylib.utils.get_project_root_dir()


# %%
class TemplateInfo(TypedDict):
    source: Literal["random", "previous", "mutated"]
    generation: int  # which round it was first created
    parent_idx: Optional[int]  # for mutated templates, track parent index


def _mutate_tmpls_with_tracking(
    tmpls_prv: th.Tensor,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    generation: int,
    parent_indices: Optional[list[int]] = None,
) -> tuple[set[tuple[int, ...]], dict[tuple[int, ...], TemplateInfo]]:
    # new candidate pool set
    fcs_set: set[tuple[int, ...]] = {
        tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls_prv
    }
    source_info: dict[tuple[int, ...], TemplateInfo] = {}
    # Tag existing templates as "previous"
    for i, _tmpl in enumerate(tmpls_prv):
        _fc = tuple(th.argwhere(_tmpl == 1).flatten().tolist())
        parent_idx = parent_indices[i] if parent_indices else i
        source_info[_fc] = TemplateInfo(
            source="previous", generation=generation - 1, parent_idx=parent_idx
        )
    # previous template pool and exclude those that has no feature to mutate from
    tmpls_prv = tmpls_prv[th.sum(tmpls_prv, dim=1) - min_features > 0]
    if len(tmpls_prv) == 0:
        return fcs_set, source_info
    # mutate templates
    pbar = tqdm.tqdm(
        itrtls.count(), desc="mutate tmpl_prv", dynamic_ncols=True, leave=False
    )
    for _c in pbar:
        if _c >= n_cands_targ or len(fcs_set) >= n_cands_targ:
            break
        # randomly choose an existing template to mutate from
        parent_tmpl_idx = int(th.randint(0, len(tmpls_prv), ()).item())
        _tmpl_prv: th.Tensor = tmpls_prv[parent_tmpl_idx]
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
            # Track as mutated template
            source_info[_fc] = TemplateInfo(
                source="mutated", generation=generation, parent_idx=parent_tmpl_idx
            )
            pbar.set_postfix({"len": len(fcs_set)})
    pbar.close()
    return fcs_set, source_info


def _fill_fcs_set_with_random_tmpls_with_tracking(
    fcs_set: set[tuple[int, ...]],
    existing_source_info: dict[tuple[int, ...], TemplateInfo],
    n_covs: int,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
    prv_featcounts: Optional[th.Tensor],
    generation: int,
) -> tuple[list[set[tuple[int, ...]]], dict[tuple[int, ...], TemplateInfo]]:
    # fill the rest of the demanded with randomly generated ones
    # group fcs_set by len(fcomb)
    max_features = n_covs if max_features is None else max_features
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [
        set() for _ in range(min_features - 1, max_features)
    ]
    [fcs_sets_by_bins[len(_fc) - min_features].add(_fc) for _fc in fcs_set]
    # Copy existing source info
    source_info = existing_source_info.copy()
    # compute maximum feature combinations allowed in each bin
    bincnt_fcs: th.Tensor = th.as_tensor(
        [
            # in order to accomondate for init_fidx,
            # both n_covs and i is one less than desired n_feats
            min(math.comb(n_covs - 1, i), th.iinfo(th.long).max)
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
        return fcs_sets_by_bins, source_info
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
                # Track as random template
                source_info[_fc] = TemplateInfo(
                    source="random", generation=generation, parent_idx=None
                )
    return fcs_sets_by_bins, source_info


def _update_template_candidates_fix_rounds_with_tracking(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    prev_template_sources: Optional[list[TemplateInfo]],
    init_fidx: int,
    n_cands_init: int,
    n_cands_mutate: int | None,
    n_cands_targ: int | None,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
    generation: int,
) -> tuple[th.Tensor, list[TemplateInfo], dict[str, int]]:
    tmpls_prv: th.Tensor = ctmpls[slctd_ms]
    n_cands_mutate = len(tmpls_prv) if n_cands_mutate is None else n_cands_mutate
    # Get parent indices for selected templates
    parent_indices = None
    if prev_template_sources:
        parent_indices = [i for i, selected in enumerate(slctd_ms) if selected]
    # Mutate templates with tracking
    fcs_set, source_info = _mutate_tmpls_with_tracking(
        tmpls_prv=tmpls_prv,
        init_fidx=init_fidx,
        n_cands_targ=min(n_cands_mutate + len(tmpls_prv), n_cands_init),
        min_features=min_features,
        generation=generation,
        parent_indices=parent_indices,
    )
    n_covs: int = ctmpls.shape[1]
    n_cands_targ = (
        5 * len(tmpls_prv) + len(fcs_set) if n_cands_targ is None else n_cands_targ
    )
    # Fill with random templates with tracking
    fcs_sets_by_bins, source_info = _fill_fcs_set_with_random_tmpls_with_tracking(
        fcs_set=fcs_set,
        existing_source_info=source_info,
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=min(n_cands_targ, n_cands_init),
        min_features=min_features,
        max_features=max_features,
        prv_featcounts=(
            th.sum(ctmpls[slctd_ms], dim=0) if use_feature_importance_sampling else None
        ),
        generation=generation,
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = len(fcs_l)
    ctmpls_new: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    template_sources: list[TemplateInfo] = []
    for _i, _fc in enumerate(fcs_l):
        ctmpls_new[_i, _fc] = 1
        template_sources.append(source_info[_fc])
    # Count sources for statistics
    source_stats = {"random": 0, "previous": 0, "mutated": 0}
    for template_info in template_sources:
        source_stats[template_info["source"]] += 1
    return ctmpls_new, template_sources, source_stats


@th.no_grad()
def make_templates_fix_rounds_with_tracking(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_init: int,
    n_cands_mutate: int | None,
    n_cands_targ: int | None,
    min_features: int,
    max_features: Optional[int],
    n_rounds: int,
    use_feature_importance_sampling: bool,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
    n_neighs: int,
    vdata: Optional[thd.TensorDict],
    metrics_func: thm.MetricCollection,
    eval_every_n_iter: int,
) -> th.Tensor:
    """Make templates using mutate greedy search with detailed source tracking.

    This function extends the standard template generation process by tracking the source
    of each template candidate (random, previous round, or mutated) and recording selection
    statistics across rounds. It provides insights into which types of templates are most
    effective and how the template population evolves.

    Args:
        tdata (thd.TensorDict): (n,) training data containing 'xs' and 'ys' keys
        max_tdata (Optional[int]): optional maximum training data to subsample if training
            data is too large and takes too long to complete. If None, uses all data.
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature
            classifier to use for template evaluation
        to_update_classifier (bool): whether to update subset feature classifier when a new
            collection of templates is generated and before the final collection of templates is found
        init_fidx (int): initial feature index that must be included in all templates
        n_tmpls_targ (int): number of templates to return in final selection
        n_cands_init (int): initial candidate pool size for round 0
        n_cands_mutate (int | None): after initial search, number of candidates coming from
            mutating previous round templates; set to `n_tmpls_targ` if `None`
        n_cands_targ (int | None): after initial search, total number of candidates in pool;
            set to `5 * n_tmpls_targ` if `None`
        min_features (int): minimum number of features to include in each template; must be >= 1
        max_features (Optional[int]): maximum number of features to include in each template;
            if `None`, `n_covs` is used
        n_rounds (int): number of mutative greedy search rounds
        use_feature_importance_sampling (bool): use feature frequencies from previous round
            identified templates to bias random candidate generation towards more successful features
        lmbda (float): penalty term for choosing more features in reward computation
        bsz (int): batch size for evaluating candidates
        plf (pl.Fabric): PyTorch Lightning fabric instance for device management and logging

    Returns:
        tuple[th.Tensor, list[dict[str, any]]]:
            - templates: (n_tmpls_targ, n_covs) final selected binary template matrix
            - selection_history: list of per-round statistics containing:
                - 'round': round number (0-indexed)
                - 'candidate_sources': dict with counts {'random': int, 'previous': int, 'mutated': int}
                - 'selected_sources': dict with counts of selected templates by source
                - 'total_candidates': total number of candidate templates in round
                - 'total_selected': number of templates selected in round

    Template Source Types:
        - 'random': Templates generated randomly (all templates in round 0, plus fill-in templates)
        - 'previous': Templates selected from the previous round, carried forward unchanged
        - 'mutated': Templates created by removing one feature from previous round selections

    Example:
        tmpls, history = make_templates_fix_rounds_with_tracking(...)

        # Print selection rates by source
        for round_stats in history:
            print(f"Round {round_stats['round']}:")
            candidates = round_stats['candidate_sources']
            selected = round_stats['selected_sources']
            for source in ['random', 'previous', 'mutated']:
                rate = selected[source] / max(candidates[source], 1) * 100
                print(f"  {source}: {selected[source]}/{candidates[source]} ({rate:.1f}%)")
    """
    classifier.eval().to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    ctmpls: th.Tensor | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    template_sources: Optional[list[TemplateInfo]] = None
    # selection_history: list[dict[str, Any]] = []
    for _i in tqdm.trange(
        n_rounds, desc="mktmpl fix rounds", leave=False, dynamic_ncols=True
    ):
        if ctmpls is None or tmpls is None or slctd_ms is None:
            # initialize candidate pool
            ctmpls = tafalib.makers.candidates.make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
            )
            # Tag all initial templates as random, generation 0
            template_sources = []
            for _tmpl in ctmpls:
                # _fc = tuple(th.argwhere(_tmpl == 1).flatten().tolist())
                template_sources.append(
                    TemplateInfo(source="random", generation=0, parent_idx=None)
                )
            source_stats = {
                "random": len(template_sources),
                "previous": 0,
                "mutated": 0,
            }
        else:
            # update candidate pool from existing templates
            ctmpls, template_sources, source_stats = (
                _update_template_candidates_fix_rounds_with_tracking(
                    ctmpls=ctmpls,
                    slctd_ms=slctd_ms,
                    prev_template_sources=template_sources,
                    init_fidx=init_fidx,
                    n_cands_init=n_cands_init,
                    n_cands_mutate=n_cands_mutate,
                    n_cands_targ=n_cands_targ,
                    min_features=min_features,
                    max_features=max_features,
                    use_feature_importance_sampling=use_feature_importance_sampling,
                    generation=_i,
                )
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
                if max_tdata is not None and max_tdata < len(tdata)
                else tdata
            ),
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        tmpls, slctd_ms = tafalib.makers.templates.make_templates_from_candidates(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"fixrounds_mktmpl{_i}",
        )
        # Track selection statistics
        selected_sources = {"random": 0, "previous": 0, "mutated": 0}
        for idx, selected in enumerate(slctd_ms):
            if selected:
                selected_sources[template_sources[idx]["source"]] += 1
        # Record round statistics
        round_stats = {
            "round": _i,
            **{f"candidates-stat-{_k}": _v for _k, _v in source_stats.items()},
            **{f"template-source-{_k}": _v for _k, _v in selected_sources.items()},
            **{
                f"selection-rate-{_k}": selected_sources[_k] / max(source_stats[_k], 1)
                for _k in source_stats.keys()
            },
            "total_candidates": len(ctmpls),
            "total_selected": int(slctd_ms.sum().item()),
        }
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(round_stats, "round-stats"), step=_i
        )
        # selection_history.append(round_stats)
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(tmpls)
        # log metrics to track progress
        if _i % eval_every_n_iter == 0:
            # _tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
            #     tmpls=tmpls,
            #     data=tdata,
            #     classifier=tclassifier,
            #     lmbda=lmbda,
            #     bsz=bsz,
            #     plf=plf,
            # )
            # tmetrics_d: dict[str, float] = tafalib.utils.evaluate(
            #     data=tdata,
            #     classifier=vclassifier,
            #     cost_est=lambda x: tafalib.functional.knn_cost_est(
            #         x,
            #         lmbda=lmbda,
            #         txs=tdata["xs"],
            #         tcels=_tpcomp["cels"],
            #         tmpls=tmpls,  # type:ignore
            #         n_neighs=n_neighs,
            #         p=2,
            #         is_train=True,
            #     ),
            #     init_fidx=init_fidx,
            #     tmpls=tmpls,
            #     metrics_func=metrics_func,
            #     plf=plf,
            # )
            # plf.log_dict(mylib.utils.add_prefix_to_dict(tmetrics_d, "train"), step=_i)
            if vdata is not None:
                _tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
                    tmpls=tmpls,
                    data=tdata,
                    classifier=tclassifier,
                    lmbda=lmbda,
                    bsz=bsz,
                    plf=plf,
                )
                vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
                    data=vdata,
                    classifier=vclassifier,
                    cost_est=lambda x: tafalib.functional.knn_cost_est(
                        x,
                        lmbda=lmbda,
                        txs=tdata["xs"],
                        tcels=_tpcomp["cels"],
                        tmpls=tmpls,  # type:ignore
                        n_neighs=n_neighs,
                        p=2,
                    ),
                    init_fidx=init_fidx,
                    tmpls=tmpls,
                    metrics_func=metrics_func,
                    plf=plf,
                )
                plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "val"), step=_i)
    assert tmpls is not None
    return tmpls


# %%
@th.no_grad()
def _rfe_mutate_tmpls(
    tdata: thd.TensorDict,
    tpcomp: thd.TensorDict,
    tmpls_prv: th.Tensor,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    rfe_kwargs: dict[str, Any],
) -> set[tuple[int, ...]]:
    assert "n_features_to_select" not in rfe_kwargs
    fcs_prv: set[tuple[int, ...]] = {
        tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls_prv
    }
    # TODO allow multiple estimators_ for an ensemble of feature selection
    # assert "estimator_" not in rfecv_kwargs
    # (n, ntmpls)
    rwds: th.Tensor = tpcomp["rwds"]
    # (n, )
    tmpl_idxs: th.Tensor = th.argmin(rwds, dim=1)
    # (n, n_covs)
    txs: th.Tensor = tdata["xs"]
    # (n, )
    tys: th.Tensor = tdata["ys"]
    newfms_l: list[th.Tensor] = list()
    for _tidx in tqdm.tqdm(
        th.unique(tmpl_idxs).tolist(),
        desc="rfe-loop",
        leave=False,
        dynamic_ncols=True,
    ):
        _idxs: th.Tensor = tmpl_idxs == _tidx
        _txs: th.Tensor = txs[_idxs, :]
        _tys: th.Tensor = tys[_idxs]
        # in case only one category in _tys
        if th.unique(_tys).numel() == 1:
            # just sample a random subset of data to perform feature selection
            _rsamp_idxs: th.Tensor = th.multinomial(
                th.ones(len(tdata)),
                num_samples=int(th.max(th.bincount(tmpl_idxs)).item()),
                replacement=True,
            )
            _txs: th.Tensor = txs[_rsamp_idxs, :]
            _tys: th.Tensor = tys[_rsamp_idxs]
        # encode label to consecutive integers starting with zero
        _tys = th.as_tensor(
            skl_preproc.LabelEncoder().fit_transform(_tys.numpy(force=True)),
            dtype=th.long,
        )
        # TODO allow multiple estimators_ for an ensemble of feature selection
        _newfms: th.Tensor = th.stack(
            [
                th.as_tensor(
                    skl_fsel.RFE(
                        estimator=xgbst.XGBClassifier(),
                        n_features_to_select=_n_features_to_select,
                        **rfe_kwargs,
                    )
                    .fit(_txs.numpy(force=True), _tys.numpy(force=True))
                    .support_,
                    dtype=th.long,
                )
                for _n_features_to_select in range(
                    min_features, int(th.sum(tmpls_prv[_tidx]).item()) + 1
                )
            ],
            dim=0,
        )
        _newfms[:, init_fidx] = 1
        newfms_l.append(_newfms)
    newfms: th.Tensor = th.unique(th.cat(newfms_l, dim=0), dim=0)
    if len(newfms) > n_cands_targ:
        newfms = newfms[
            th.multinomial(
                th.ones(len(newfms)), num_samples=n_cands_targ, replacement=False
            )
        ]
    fcs_set: set[tuple[int, ...]] = {
        tuple(sorted(th.argwhere(_nfm == 1).flatten().tolist())) for _nfm in newfms
    }
    fcs_set = {_fc for _fc in fcs_set if _fc not in fcs_prv}
    return fcs_set


@th.no_grad()
def _fine_tune_templates_with_rfe(
    tdata: thd.TensorDict,
    tmpls: th.Tensor,  # Templates from make_templates_fix_rounds_with_tracking
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    rfe_kwargs: dict[str, Any],
    min_features: int,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
    n_neighs: int,
    vdata: Optional[thd.TensorDict],
    metrics_func: thm.MetricCollection,
) -> th.Tensor:
    """Fine-tune templates using RFE (Recursive Feature Elimination).

    Takes templates generated by make_templates_fix_rounds_with_tracking and applies
    RFE to create refined versions with better feature selection.

    Args:
        tdata: Training data with 'xs' and 'ys' keys
        tmpls: Input templates to fine-tune (n_tmpls, n_covs)
        classifier: Subset feature classifier for evaluation
        init_fidx: Initial feature index that must be preserved
        rfe_kwargs: RFE parameters (e.g., {'step': 0.03})
        min_features: Minimum number of features per template
        lmbda: Penalty coefficient for template complexity
        bsz: Batch size for evaluation
        plf: PyTorch Lightning Fabric instance

    Returns:
        Fine-tuned templates tensor
    """
    assert "n_features_to_select" not in rfe_kwargs
    # Precompute rewards for input templates
    tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
        tmpls=tmpls,
        data=tdata,
        classifier=classifier,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )
    # log metrics to track progress
    if vdata is not None:
        vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
            data=vdata,
            classifier=classifier,
            cost_est=lambda x: tafalib.functional.knn_cost_est(
                x,
                lmbda=lmbda,
                txs=tdata["xs"],
                tcels=tpcomp["cels"],
                tmpls=tmpls,  # type:ignore
                n_neighs=n_neighs,
                p=2,
            ),
            init_fidx=init_fidx,
            tmpls=tmpls,
            metrics_func=metrics_func,
            plf=plf,
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "pre-finetune"), step=0)
    # Apply RFE to generate refined templates
    refined_fcs_set: set[tuple[int, ...]] = _rfe_mutate_tmpls(
        tdata=tdata,
        tpcomp=tpcomp,
        tmpls_prv=tmpls,
        init_fidx=init_fidx,
        n_cands_targ=len(tmpls) * 3,  # Generate more candidates for selection
        min_features=min_features,
        rfe_kwargs=rfe_kwargs,
    )
    # Add original templates to the candidate pool
    original_fcs_set: set[tuple[int, ...]] = {
        tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls
    }
    all_fcs_set: set[tuple[int, ...]] = refined_fcs_set.union(original_fcs_set)
    # Convert feature combinations back to template format
    n_covs: int = tmpls.shape[1]
    refined_tmpls: th.Tensor = th.zeros((len(all_fcs_set), n_covs), dtype=th.long)
    for i, fc in enumerate(all_fcs_set):
        refined_tmpls[i, fc] = 1
    # Evaluate refined templates and select the best ones
    refined_tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
        tmpls=refined_tmpls,
        data=tdata,
        classifier=classifier,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )
    # Select the same number of templates as input
    final_tmpls, final_slctd_ms = (
        tafalib.makers.templates.make_templates_from_candidates(
            tpcomp=refined_tpcomp,
            ctmpls=refined_tmpls,
            n_tmpls=len(tmpls),
            plf=plf,
            log_prefix="rfe_finetune",
        )
    )
    final_tpcomp: thd.TensorDict = thd.TensorDict(
        {_k: _v[:, final_slctd_ms] for _k, _v in refined_tpcomp.items()}
    ).auto_batch_size_(1)
    # log metrics to track progress
    if vdata is not None:
        vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
            data=vdata,
            classifier=classifier,
            cost_est=lambda x: tafalib.functional.knn_cost_est(
                x,
                lmbda=lmbda,
                txs=tdata["xs"],
                tcels=final_tpcomp["cels"],
                tmpls=tmpls,  # type:ignore
                n_neighs=n_neighs,
                p=2,
            ),
            init_fidx=init_fidx,
            tmpls=tmpls,
            metrics_func=metrics_func,
            plf=plf,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(vmetrics_d, "post-finetune"), step=0
        )
    return final_tmpls


# %%
# NOTE mnist
data_name: str = "mnist"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=True)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
max_tdata: Optional[int] = 30000
_tdata_shuffle_idxs = th.randperm(
    len(_tdata), generator=th.Generator().manual_seed(279)
)
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
run_p: str = (
    "experiments/pretrain/nnet_subset_feature_classifier/outputs/mnist/20250312_225510"
)
run_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, run_p, ".hydra", "config.yaml"))
tclassifier = _classifiers.make_concat_nnet_classifier_from_pretrain_run(
    run_p=run_p,
    xs_train=tdata["xs"].numpy(),
    ys_train=tdata["ys"].numpy(),
    fit_kwargs=hd.utils.instantiate(run_cfg.nnet_fit_cfg),
)
vclassifier = tclassifier
init_fidx: int = 100
n_tmpls_targ: int = 512
n_cannds_init: int = 10_000
n_cands_targ: int = 10_000
n_cands_mutate: int = 2048
lmbda: float = 0.055
n_neighs = 100
min_features_targ: int = 1
max_features_targ: Optional[int] = 32
n_rounds: int = 10
use_feature_importance_sampling: bool = True
to_update_classifier: bool = False
bsz: int = 61440

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "mutate-then-select")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)

# %%
if init_fidx is None:
    init_fidx, bestfm = tafalib.makers.templates.identify_init_fidx(
        tdata=tdata,
        classifier=tclassifier,
        max_features=max_features_targ,
        n_repeat=2,
        n_masks=500,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )


# %%
tmpls = make_templates_fix_rounds_with_tracking(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    to_update_classifier=to_update_classifier,
    init_fidx=init_fidx,
    n_tmpls_targ=n_tmpls_targ,
    n_cands_init=n_cands_targ,
    n_cands_mutate=n_cands_mutate,
    n_cands_targ=n_cands_targ,
    min_features=min_features_targ,
    max_features=max_features_targ,
    n_rounds=n_rounds,
    use_feature_importance_sampling=use_feature_importance_sampling,
    lmbda=lmbda,
    bsz=bsz,
    plf=plf,
    n_neighs=n_neighs,
    vdata=vdata,
    metrics_func=metrics_func,
    eval_every_n_iter=1,
)

# %%
post_tmpls: th.Tensor = _fine_tune_templates_with_rfe(
    tdata=tdata,
    tmpls=tmpls,
    classifier=tclassifier,
    init_fidx=init_fidx,
    rfe_kwargs={
        "step": 0.03,
    },
    min_features=min_features_targ,
    lmbda=lmbda,
    bsz=bsz,
    plf=plf,
    n_neighs=n_neighs,
    vdata=vdata,
    metrics_func=metrics_func,
)


# %%
tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
    tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz, plf=plf
)

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=n_neighs,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
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
        "init_fidx": init_fidx,
        "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
        "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
    }
)
print(pd.Series(metrics_d))

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=tclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=n_neighs,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
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
        "init_fidx": init_fidx,
        "feature used & observed": th.mean(
            th.as_tensor(snfobsd_l, dtype=th.float32)
        ).item(),
    }
)
print(pd.Series(metrics_d))

# %%
