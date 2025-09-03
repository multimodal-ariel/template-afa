# %%
from __future__ import annotations

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
import tafalib.functional
import tafalib.makers.templates
import tafalib.utils
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst
import sklearn.preprocessing as skl_preproc
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


def _update_template_candidates_rfe_with_tracking(
    tdata: thd.TensorDict,
    tpcomp: thd.TensorDict,
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    prev_template_sources: Optional[list[TemplateInfo]],
    init_fidx: int,
    rfe_kwargs: dict[str, Any],
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
    # new candidate pool set
    fcs_set: set[tuple[int, ...]] = {
        tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls_prv
    }
    # Get parent indices for selected templates
    parent_indices = None
    if prev_template_sources:
        parent_indices = [i for i, selected in enumerate(slctd_ms) if selected]
    source_info: dict[tuple[int, ...], TemplateInfo] = {}
    # Tag existing templates as "previous"
    for i, _tmpl in enumerate(tmpls_prv):
        _fc = tuple(th.argwhere(_tmpl == 1).flatten().tolist())
        parent_idx = parent_indices[i] if parent_indices else i
        source_info[_fc] = TemplateInfo(
            source="previous", generation=generation - 1, parent_idx=parent_idx
        )
    # Mutate templates with tracking
    new_fcs_set = _rfe_mutate_tmpls(
        tdata=tdata,
        tpcomp=tpcomp,
        tmpls_prv=tmpls_prv,
        init_fidx=init_fidx,
        n_cands_targ=n_cands_mutate,
        min_features=min_features,
        rfe_kwargs=rfe_kwargs,
    )
    fcs_set.update(new_fcs_set)
    source_info.update(
        {
            _fc: TemplateInfo(source="mutated", generation=generation, parent_idx=None)
            for _fc in new_fcs_set
        }
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
def make_templates_rfe_with_tracking(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: int,
    n_tmpls_targ: int,
    rfe_kwargs: dict[str, Any],
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
    """Generate feature subset templates using RFE-based evolutionary search with source tracking.

    This function implements an evolutionary template generation algorithm that combines multiple
    strategies for creating feature subset candidates: random generation, template carryover from
    previous rounds, and intelligent mutation using Recursive Feature Elimination (RFE). Each
    template's source and generation are tracked to provide insights into which strategies are
    most effective during the evolutionary process.

    The algorithm operates in iterative rounds:
    1. Initial round generates random template candidates
    2. Subsequent rounds carry forward successful templates, mutate them using RFE, and fill
       remaining slots with new random templates
    3. All candidates are evaluated and the best subset is selected for the next round
    4. Selection statistics by source type are logged for analysis

    Args:
        tdata (thd.TensorDict): Training data containing 'xs' (features) and 'ys' (labels) keys
        max_tdata (Optional[int]): Maximum number of training samples to use for evaluation.
            If None or larger than available data, uses all training data. Useful for reducing
            computation time with large datasets.
        classifier (mymodels.classifiers.SubsetFeatureClassifier): Pretrained subset feature
            classifier used to evaluate template performance
        to_update_classifier (bool): Whether to retrain/update the classifier when new template
            collections are generated. Only applicable for SubsetFeatureConcatClassifier.
        init_fidx (int): Index of the initial feature that must be included in all templates.
            This feature serves as a required baseline.
        n_tmpls_targ (int): Number of templates to select and return in the final collection
        rfe_kwargs (dict[str, Any]): Keyword arguments passed to sklearn's RFE for template
            mutation. Common keys include 'step' (fraction of features to remove per iteration).
            Note: 'n_features_to_select' and 'estimator' are handled internally.
        n_cands_init (int): Size of initial candidate pool in round 0 (all random templates)
        n_cands_mutate (int | None): Number of mutated template candidates to generate from
            selected templates in each round after round 0. If None, uses n_tmpls_targ.
        n_cands_targ (int | None): Total target size of candidate pool in rounds after round 0.
            Remaining slots after previous templates and mutations are filled with random
            templates. If None, uses 5 * n_tmpls_targ + existing templates.
        min_features (int): Minimum number of features per template (must be >= 1)
        max_features (Optional[int]): Maximum number of features per template. If None, uses
            total number of available features (n_covs).
        n_rounds (int): Number of evolutionary rounds to perform
        use_feature_importance_sampling (bool): Whether to bias random template generation
            toward features that appeared frequently in previously selected templates
        lmbda (float): Penalty coefficient for template complexity in reward computation.
            Higher values favor templates with fewer features.
        bsz (int): Batch size for template evaluation during reward computation
        plf (pl.Fabric): PyTorch Lightning Fabric instance for device management and logging

    Returns:
        th.Tensor: Selected templates as binary matrix of shape (n_tmpls_targ, n_covs) where
            each row represents a template and 1s indicate selected features.

    Template Source Tracking:
        The function tracks three types of template sources:
        - 'random': Newly generated random templates (all templates in round 0, plus fill-ins)
        - 'previous': Templates carried forward from the previous round's selection
        - 'mutated': Templates created by applying RFE to remove features from previous selections

        Statistics for each source type (candidate counts, selection counts, selection rates) are
        logged per round using the provided Fabric logger with 'round-stats' prefix.

    RFE Mutation Process:
        For each selected template from the previous round:
        1. Identify training samples where this template performed best
        2. Apply RFE with XGBoost on these samples to generate feature importance rankings
        3. Create multiple mutation candidates by removing different numbers of features
        4. Ensure the required init_fidx feature is preserved in all mutations
        5. Select up to n_cands_mutate unique mutations for the candidate pool

    Example:
        ```python
        templates = make_templates_rfe_with_tracking(
            tdata=train_data,
            max_tdata=10000,
            classifier=my_classifier,
            to_update_classifier=True,
            init_fidx=42,
            n_tmpls_targ=256,
            rfe_kwargs={'step': 0.1},
            n_cands_init=5000,
            n_cands_mutate=1000,
            n_cands_targ=5000,
            min_features=3,
            max_features=20,
            n_rounds=5,
            use_feature_importance_sampling=True,
            lmbda=0.01,
            bsz=1024,
            plf=fabric_instance
        )
        ```

    Note:
        Selection statistics are automatically logged to the provided Fabric loggers with keys
        like 'round-stats/candidates-stat-random', 'round-stats/selection-rate-mutated', etc.
    """
    classifier.eval().to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    ctmpls: th.Tensor | None = None
    tpcomp: thd.TensorDict | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    template_sources: Optional[list[TemplateInfo]] = None
    # selection_history: list[dict[str, Any]] = []
    for _i in tqdm.trange(
        n_rounds, desc="mktmpl fix rounds", leave=False, dynamic_ncols=True
    ):
        _generator: th.Generator = (
            th.Generator().manual_seed(279) if _i == 0 else th.default_generator
        )
        if ctmpls is None or tmpls is None or slctd_ms is None or tpcomp is None:
            # initialize candidate pool
            ctmpls = tafalib.makers.candidates.make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
                # NOTE: fix initial candidates
                generator=_generator,
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
                _update_template_candidates_rfe_with_tracking(
                    tdata=tdata,
                    tpcomp=tafalib.utils.precomp_rwds_for_tmpls(
                        tmpls=tmpls,
                        data=tdata,
                        classifier=classifier,
                        lmbda=lmbda,
                        bsz=bsz,
                        plf=plf,
                    ),
                    ctmpls=ctmpls,
                    slctd_ms=slctd_ms,
                    prev_template_sources=template_sources,
                    init_fidx=init_fidx,
                    rfe_kwargs=rfe_kwargs,
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
        tpcomp = tafalib.utils.precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[
                    th.multinomial(
                        th.ones((len(tdata),)),
                        num_samples=max_tdata,
                        generator=_generator,
                    )
                ]
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
                    lmbda=lmbda,
                    metrics_func=metrics_func,
                    plf=plf,
                )
                plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "val"), step=_i)
    assert tmpls is not None
    return tmpls


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
output_dir: str = os.path.join(
    "outputs", "run", data_name, "rfe-candidates-selection-rate"
)
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
tmpls = make_templates_rfe_with_tracking(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    to_update_classifier=to_update_classifier,
    init_fidx=init_fidx,
    rfe_kwargs={
        "step": 0.03,
    },
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
