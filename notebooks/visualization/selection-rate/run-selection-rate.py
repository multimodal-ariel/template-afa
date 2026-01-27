# %%
from __future__ import annotations

import itertools as itrtls
import math
import os
from typing import Literal, Optional, TypedDict

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels.classifiers
import tafalib.functional
import tafalib.makers.templates
import tafalib.utils
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
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
        _generator: th.Generator = (
            th.Generator().manual_seed(279) if _i == 0 else th.default_generator
        )
        if ctmpls is None or tmpls is None or slctd_ms is None:
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
            {"round": _i, **mylib.utils.add_prefix_to_dict(round_stats, "round-stats")},
            step=_i,
        )
        # selection_history.append(round_stats)
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(tmpls)
        # log metrics to track progress
        if (_i % eval_every_n_iter) == 0:
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
                plf.log_dict(
                    {"round": _i, **mylib.utils.add_prefix_to_dict(vmetrics_d, "val")},
                    step=_i,
                )
    assert tmpls is not None
    return tmpls


# %%
# # NOTE mnist
# data_name: str = "mnist"
# _tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=True)
# n_covs: int = _tdata["xs"].shape[1]
# n_labels: int = len(th.unique(_tdata["ys"]))
# max_tdata: Optional[int] = 30000
# _tdata_shuffle_idxs = th.randperm(
#     len(_tdata), generator=th.Generator().manual_seed(279)
# )
# tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
# extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
run_p: str = "experiments/make_template/outputs/big5_cnnet/20251213_024429/0"
run_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, run_p, ".hydra", "config.yaml"))
tafa_cfg = OmegaConf.load(
    os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
)  # type:ignore
data_name: str = tafa_cfg.data.name
_tdata, vdata, tstdata = hd.utils.call(tafa_cfg.data)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
# split training data into two for classifier and afa
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata: thd.TensorDict
extdata: thd.TensorDict
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(mylib.utils.get_project_root_dir(), run_p, "tdata_shuffle_idxs.pt")
)
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
# make classifier
tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
    tafa_cfg.tclassifier,
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
)
if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
    tclassifier_p = os.path.join(
        mylib.utils.get_project_root_dir(), run_p, "tclassifier.pt"
    )
    if os.path.exists(tclassifier_p):
        tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
    else:
        tclassifier.fit_(tmpls)
vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
if tafa_cfg.vclassifier is not None:
    vclassifier = hd.utils.instantiate(
        tafa_cfg.vclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )
    if isinstance(vclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        vclassifier_p = os.path.join(
            mylib.utils.get_project_root_dir(), run_p, "tclassifier.pt"
        )
        if os.path.exists(vclassifier_p):
            vclassifier.load_state_dict(th.load(vclassifier_p, map_location="cpu"))
        else:
            vclassifier.fit_(tmpls)

# %%
init_fidx: int = tafa_cfg.init_fidx
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
bsz: int = 128000

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
    max_tdata=None,
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
