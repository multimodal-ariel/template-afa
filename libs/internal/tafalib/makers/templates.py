from __future__ import annotations

import copy
import itertools as itrtls
import math
import os
from abc import abstractmethod
from typing import Callable, Generator, Iterable, Optional, Protocol

import lightning as pl
import mylib
import mymodels
import scipy.spatial as sp_spatial
import tensordict as thd
import torch as th
import torch.distributions.utils
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm

from .. import functional as tafalib_functional
from .. import utils as tafalib_utils
from . import candidates as tafalib_makers_candidates


@th.no_grad()
def identify_init_fidx(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    max_features: Optional[int],
    n_repeat: int,
    n_masks: int,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> tuple[int, th.Tensor]:
    """a function to find initial feature index

    Args:
        tdata (thd.TensorDict): (n, ) training data and must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature classifier
        max_features (Optional[int]): optional maximum number of features; if `none`, no restriction on number of features included in template.
        n_repeat (int): number of trials to search for initial feature
        n_masks (int): number of candidates to condsider for each trail
        lmbda (float): penalty term for choosing more features
        bsz (int): batch size

    Returns::
        int: initial feature index
        th.Tensor: (n_covs, ) the costs of each feature
    """
    best_fms: th.Tensor = th.stack(
        [
            _ident_init_fidx_single(
                tdata=tdata,
                classifier=classifier,
                max_features=max_features,
                n_masks=n_masks,
                lmbda=lmbda,
                bsz=bsz,
                plf=plf,
            )[1]
            for _ in tqdm.trange(
                n_repeat, desc="ident init_fidx", leave=False, dynamic_ncols=True
            )
        ]
    )
    best_fm: th.Tensor = th.mean(best_fms, dim=0)
    return int(th.argmin(best_fm)), best_fm


class _PostHocIdentifyInitialFeatureFunc(Protocol):
    @abstractmethod
    def __call__(
        self,
        tdata: thd.TensorDict,
        tmpls: th.Tensor,
        tpcomp: Optional[thd.TensorDict],
        classifier: mymodels.classifiers.SubsetFeatureClassifier,
        lmbda: float,
        bsz: int,
        plf: pl.Fabric,
    ) -> tuple[int, thd.TensorDict]:
        pass


def post_hoc_count_based_identify_init_fidx(
    tdata: thd.TensorDict,
    tmpls: th.Tensor,
    tpcomp: Optional[thd.TensorDict],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> tuple[int, thd.TensorDict]:
    assert tpcomp is not None
    n_tmpls: int = len(tmpls)
    assert tpcomp.shape[1] == n_tmpls
    # (n, n_tmpls)
    rwds: th.Tensor = tpcomp["rwds"]
    # (n, )
    max_rwds, max_tmpl_idxs = th.max(rwds, dim=1)
    # (n, n_covs)
    fms: th.Tensor = tmpls[max_tmpl_idxs]
    fcounts: th.Tensor = th.sum(fms, dim=0)
    sorted_fcounts, sorted_fidxs = th.sort(fcounts, descending=True)
    infos: thd.TensorDict = thd.make_tensordict(
        {
            "max_rwds": max_rwds,
            "max_tmpl_idxs": max_tmpl_idxs,
            "sorted_fcounts": sorted_fcounts,
            "sorted_fidxs": sorted_fidxs,
        }
    ).auto_device_()
    return int(sorted_fidxs[0].item()), infos


@th.no_grad()
def post_cost_try_then_identify_init_fidx(
    tdata: thd.TensorDict,
    tmpls: th.Tensor,
    tpcomp: Optional[thd.TensorDict],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> tuple[int, thd.TensorDict]:
    txs: th.Tensor = tdata["xs"]
    n_covs: int = txs.shape[1]
    best_mean_rwd: float = -th.inf
    best_init_fidx: int = -1
    pbar = tqdm.trange(n_covs, desc="try-init-feat", dynamic_ncols=True, leave=False)
    for _fidx in pbar:
        _tmpls: th.Tensor = tmpls.clone()
        _tmpls[:, _fidx] = 1
        _tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            tmpls=_tmpls,
            data=tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        _rwds: th.Tensor = _tpcomp["rwds"]
        _mean_rwd: float = th.mean(th.max(_rwds, dim=1)[0], dim=0).item()
        if _mean_rwd > best_mean_rwd:
            best_mean_rwd = _mean_rwd
            best_init_fidx = _fidx
        pbar.set_postfix(
            {
                "mean-rwd": _mean_rwd,
                "best-mean-rwd": best_mean_rwd,
                "best-init-fidx": best_init_fidx,
            }
        )
    pbar.close()
    infos = thd.make_tensordict({"best_mean_rwd": th.as_tensor(best_mean_rwd)})
    return best_init_fidx, infos


# NOTE make templates from scratch
@th.no_grad()
def make_templates_vanilla(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: Optional[int | _PostHocIdentifyInitialFeatureFunc],
    n_tmpls: int,
    n_cands: int,
    min_features: int,
    max_features: Optional[int],
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """make templates greedily

    Args:
        tdata (thd.TensorDict): (n, ) training data
        max_tdata (Optional[int]): optional maximum training data to subsample if training data is too large and takes too long to complete
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature classifier to use
        to_update_classifier (bool): to update subset feature classifier when a new collection of templates is generated and before the final collection of templates is found.
        init_fidx (int): initial feature index
        n_tmpls (int): number of templates to return
        n_cands (int): number of candidates used to search
        min_features (int): minimum number of features to include in each template; must be greater than 1.
        max_features (Optional[int]): maximum number of features to include in each template; if `None`, `n_covs` is used.
        lmbda (float): penalty term for choosing more features.
        bsz (int): batch size for evaluating candidates
        plf (pl.Fabric): lightning fabric instance

    Returns:
        th.Tensor: (n_tmpls, n_covs)
    """
    generator = th.default_generator if generator is None else generator
    classifier.to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # NOTE init. candidate templates
    ctmpls: th.Tensor = (
        tafalib_makers_candidates.make_template_candidates(
            n_covs=n_covs,
            init_fidx=init_fidx,
            n_cands_targ=n_cands,
            min_features=min_features,
            max_features=max_features,
            generator=generator,
        )
        if init_fidx is not None and isinstance(init_fidx, int)
        else tafalib_makers_candidates.make_feature_masks(
            n_covs=n_covs,
            n_masks=n_cands,
            min_features=min_features,
            max_features=max_features,
            generator=generator,
        )
    )
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(ctmpls)
    tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
        tmpls=ctmpls,
        data=(
            tdata[
                th.multinomial(
                    th.ones((len(tdata),)), num_samples=max_tdata, generator=generator
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
    tmpls, slctd_ms = make_templates_from_candidates(
        tpcomp=tpcomp,
        ctmpls=ctmpls,
        n_tmpls=n_tmpls,
        plf=plf,
        log_prefix="vanilla_mktmpl",
    )
    tpcomp.auto_batch_size_(2)
    tpcomp_slctd: thd.TensorDict = tpcomp[:, slctd_ms]
    if init_fidx is None:
        init_fidx = post_hoc_count_based_identify_init_fidx
    if not isinstance(init_fidx, int):
        fidx = init_fidx(
            tdata=tdata,
            tmpls=tmpls,
            tpcomp=tpcomp_slctd,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )[0]
        tmpls[:, fidx] = 1
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(tmpls)
    return tmpls


@th.no_grad()
def make_greedy_correlation_templates(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: int,
    n_tmpls: int,
    n_cands: int,
    min_features: int,
    max_features: Optional[int],
    make_candidates_fn: Callable[
        [th.Tensor, int, int, int, int], th.Tensor
    ],  # TODO: might need to be changed
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """
    Generates correlation-based templates using a classifier and candidate template generator.
    This function creates candidate templates from input data, optionally updates the classifier,
    precomputes rewards for the templates, selects the best templates, and returns them. It supports
    limiting the number of data samples used, updating the classifier before and after template selection,
    and customizing candidate generation.

    Args:
        tdata (thd.TensorDict): Input data as a TensorDict containing features and possibly labels.
        max_tdata (Optional[int]): Maximum number of data samples to use. If None, use all data.
        classifier (mymodels.classifiers.SubsetFeatureClassifier): Classifier used for template evaluation.
        to_update_classifier (bool): Whether to update (fit) the classifier with candidate/final templates.
        init_fidx (int): Initial feature index for candidate generation.
        n_tmpls (int): Number of templates to select and return.
        n_cands (int): Number of candidate templates to generate.
        min_features (int): Minimum number of features per candidate template.
        max_features (Optional[int]): Maximum number of features per candidate template. If None, uses classifier's n_covs.
        make_candidates_fn (Callable[[th.Tensor, int, int, int, int], th.Tensor]): Function to generate candidate templates.
        lmbda (float): Regularization or weighting parameter for reward computation.
        bsz (int): Batch size for processing.
        plf (pl.Fabric): Fabric object specifying device and distributed settings.

    Returns:
        th.Tensor: Selected templates as a tensor.
    """
    generator = th.default_generator if generator is None else generator
    classifier.to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # NOTE init. candidate templates
    # TODO might need to change
    ctmpls: th.Tensor = make_candidates_fn(
        xs=tdata["xs"],
        init_fidx=init_fidx,
        n_cands_targ=n_cands,
        min_features=min_features,
        max_features=max_features,
    )
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(ctmpls)
    tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
        tmpls=ctmpls,
        data=(
            tdata[
                th.multinomial(
                    th.ones((len(tdata),)), num_samples=max_tdata, generator=generator
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
    tmpls, slctd_ms = make_templates_from_candidates(
        tpcomp=tpcomp,
        ctmpls=ctmpls,
        n_tmpls=n_tmpls,
        plf=plf,
        log_prefix="corr-mktmpl",
    )
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(tmpls)
    return tmpls


@th.no_grad()
def make_templates_reduce_features(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
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
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    generator = th.default_generator if generator is None else generator
    classifier.to(device=plf.device)
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
            ctmpls = tafalib_makers_candidates.make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_targ,
                min_features=_minfeats,
                max_features=_maxfeats,
                generator=generator,
            )
        else:
            # update candidate pool from existing templates
            _maxfeats = min(max_features_targ, int(th.max(th.sum(tmpls, dim=1)).item()))
            ctmpls = _update_template_candidates(
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_targ,
                min_features=_minfeats,
                max_features=_maxfeats,
                use_feature_importance_sampling=use_feature_importance_sampling,
                generator=generator,
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[
                    th.multinomial(
                        th.ones((len(tdata),)),
                        num_samples=max_tdata,
                        generator=generator,
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
        tmpls, slctd_ms = make_templates_from_candidates(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"reduce_mktmpl{_i}",
        )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(tmpls)
        _i = _i + 1
    assert tmpls is not None
    return tmpls


@th.no_grad()
def make_templates_fix_rounds(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: Optional[int | _PostHocIdentifyInitialFeatureFunc],
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
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """make templates using mutate greedy search

    Args:
        tdata (thd.TensorDict): (n, ) training data
        max_tdata (Optional[int]): optional maximum training data to subsample if training data is too large and takes too long to complete
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature classifier to use
        to_update_classifier (bool): _descriptto update subset feature classifier when a new collection of templates is generated and before the final collection of templates is found.
        init_fidx (int): initial feature index
        n_tmpls_targ (int): number of templates to return
        n_cands_init (int): initial candidate size
        n_cands_mutate (int | None): after initial search, number of candidates coming from mutating previous round templates; set to `n_tmpls_targ` if `None`
        n_cands_targ (int | None): after initial search, number of toal amount of candidates; set to `5 * n_tmpls_targ` if `None`.
        min_features (int): minimum number of features to include in each template; must be greater than 1.
        max_features (Optional[int]): maximum number of features to include in each template; if `None`, `n_covs` is used.
        n_rounds (int): number of mutative greedy search
        use_feature_importance_sampling (bool): use feautre frequencies from previous round identified templates to sample candidates for next round of search.
        lmbda (float): penalty term for choosing more features.
        bsz (int): batch size for evaluating candidates
        plf (pl.Fabric): lightning fabric instance

    Returns:
        th.Tensor: (n_tmpls, n_covs)
    """
    generator = th.default_generator if generator is None else generator
    classifier.eval().to(device=plf.device)
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
            ctmpls = (
                tafalib_makers_candidates.make_template_candidates(
                    n_covs=n_covs,
                    init_fidx=init_fidx,
                    n_cands_targ=n_cands_init,
                    min_features=min_features,
                    max_features=max_features,
                    generator=generator,
                )
                if init_fidx is not None and isinstance(init_fidx, int)
                else tafalib_makers_candidates.make_feature_masks(
                    n_covs=n_covs,
                    n_masks=n_cands_init,
                    min_features=1,
                    max_features=max_features,
                    generator=generator,
                )
            )
        else:
            # update candidate pool from existing templates
            ctmpls = _update_template_candidates_fix_rounds(
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                init_fidx=init_fidx,
                n_cands_init=n_cands_init,
                n_cands_mutate=n_cands_mutate,
                n_cands_targ=n_cands_targ,
                min_features=min_features,
                max_features=max_features,
                use_feature_importance_sampling=use_feature_importance_sampling,
                generator=generator,
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[
                    th.multinomial(
                        th.ones((len(tdata),)),
                        num_samples=max_tdata,
                        generator=generator,
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
        tmpls, slctd_ms = make_templates_from_candidates(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"fixrounds_mktmpl{_i}",
        )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(tmpls)
    assert tmpls is not None
    tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
        tmpls,
        data=(
            tdata[
                th.multinomial(
                    th.ones((len(tdata),)),
                    num_samples=max_tdata,
                    generator=generator,
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
    if init_fidx is None:
        init_fidx = post_hoc_count_based_identify_init_fidx
    if not isinstance(init_fidx, int):
        fidx = init_fidx(
            tdata=tdata,
            tmpls=tmpls,
            tpcomp=tpcomp,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )[0]
        tmpls[:, fidx] = 1
    return tmpls


def make_templates_fix_rounds_minibatch(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_targ: int,
    n_cands_targ_minibatch: int,
    min_features: int,
    max_features: Optional[int],
    n_rounds: int,
    use_feature_importance_sampling: bool,
    lmbda: float,
    bsz: int,
    minibatch_size: int,
    plf: pl.Fabric,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    classifier.eval().to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    ctmpls: th.Tensor | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    for _i in tqdm.trange(
        n_rounds, desc="mktmpl fix rounds", leave=False, dynamic_ncols=True
    ):
        if ctmpls is None or tmpls is None or slctd_ms is None:
            if isinstance(
                classifier, mymodels.classifiers.SubsetFeatureConcatClassifier
            ):
                classifier.fit_(
                    tafalib_makers_candidates.make_template_candidates(
                        n_covs=n_covs,
                        init_fidx=init_fidx,
                        n_cands_targ=max(n_cands_targ, n_cands_targ_minibatch),
                        min_features=min_features,
                        max_features=max_features,
                        generator=generator,
                    )
                )
            ctmpls = th.cat(
                [
                    make_templates_vanilla(
                        tdata=tdata[
                            th.multinomial(
                                th.ones((len(tdata),)),
                                num_samples=minibatch_size,
                                generator=generator,
                            )
                        ],
                        max_tdata=None,
                        classifier=classifier,
                        to_update_classifier=False,
                        init_fidx=init_fidx,
                        n_tmpls=n_tmpls_targ,
                        n_cands=n_cands_targ_minibatch,
                        min_features=min_features,
                        max_features=max_features,
                        lmbda=lmbda,
                        bsz=bsz,
                        plf=plf,
                        generator=generator,
                    )
                    for _ in range(math.ceil(n_cands_targ / n_tmpls_targ))
                ],
                dim=0,
            )
        else:
            _ctmpls_prv: th.Tensor = ctmpls
            # update candidate pool from existing templates
            _ctmpls_l: list[th.Tensor] = [
                _update_template_candidates_fix_rounds(
                    ctmpls=_ctmpls_prv,
                    slctd_ms=slctd_ms,
                    init_fidx=init_fidx,
                    n_cands_init=n_cands_targ_minibatch,
                    n_cands_mutate=None,
                    n_cands_targ=None,
                    min_features=min_features,
                    max_features=max_features,
                    use_feature_importance_sampling=use_feature_importance_sampling,
                    generator=generator,
                )
                for _ in range(math.ceil(n_cands_targ / n_tmpls_targ))
            ]
            if isinstance(
                classifier, mymodels.classifiers.SubsetFeatureConcatClassifier
            ):
                classifier.fit_(th.unique(th.cat(_ctmpls_l, dim=0), dim=0))
            ctmpls = th.cat(
                [
                    make_templates_from_candidates(
                        tpcomp=tafalib_utils.precomp_rwds_for_tmpls(
                            tmpls=_ctmpls,
                            data=tdata[
                                th.multinomial(
                                    th.ones((len(tdata),)),
                                    num_samples=minibatch_size,
                                    generator=generator,
                                )
                            ],
                            classifier=classifier,
                            lmbda=lmbda,
                            bsz=bsz,
                            plf=plf,
                        ),
                        ctmpls=_ctmpls,
                        n_tmpls=n_tmpls_targ,
                        plf=plf,
                        log_prefix=f"round{_i}_minibatch{_j}",
                    )[0]
                    for _j, _ctmpls in enumerate(_ctmpls_l)
                ],
                dim=0,
            )
        ctmpls = th.unique(ctmpls, dim=0)
        assert ctmpls is not None
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[
                    th.multinomial(
                        th.ones((len(tdata),)),
                        num_samples=max_tdata,
                        generator=generator,
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
    to_update_classifier: bool,
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
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    generator = th.default_generator if generator is None else generator
    classifier.eval().to(device=plf.device)
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
            ctmpls = tafalib_makers_candidates.make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
                generator=generator,
            )
        else:
            # update candidate pool from existing templates
            ctmpls = _update_template_candidates_fix_rounds(
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                init_fidx=init_fidx,
                n_cands_init=n_cands_init,
                n_cands_mutate=None,
                n_cands_targ=None,
                min_features=min_features,
                max_features=max_features,
                use_feature_importance_sampling=use_feature_importance_sampling,
                generator=generator,
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        _tdata = (
            tdata[
                th.multinomial(
                    th.ones((len(tdata),)), num_samples=max_tdata, generator=generator
                )
            ]
            if max_tdata is not None and max_tdata < len(tdata)
            else tdata
        )
        tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            ctmpls, data=_tdata, classifier=classifier, lmbda=lmbda, bsz=bsz, plf=plf
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
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(tmpls)
    assert tmpls is not None
    return tmpls


@th.enable_grad()
def make_vanilla_gradient_descent_templates(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier,
    to_update_classifier: bool,
    init_fidx: int,
    n_tmpls: int,
    n_tdata_minibatch: int,
    n_cands_minibatch: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
    lmbda: float,
    bsz: int,
    make_opt_fn: Callable[[Iterable[th.Tensor]], th.optim.Optimizer],
    n_gradient_mutate_iters: int,
    n_gradient_steps_per_mutate_iter: int,
    plf: pl.Fabric,
    generator: Optional[th.Generator] = None,
):
    """
    Generates a set of high-quality template feature selectors using gradient descent.

    This function uses a neural network classifier and gradient-based optimization to iteratively
    mutate and select binary feature templates. The process involves generating candidate templates,
    optimizing them via gradient descent to maximize a reward (classification performance penalized
    by template size), and selecting the best templates according to their precomputed rewards.

    Args:
        tdata (thd.TensorDict): Training data containing features and labels.
        max_tdata (Optional[int]): Maximum number of training data points to use for template evaluation.
        classifier (mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier): Classifier used for reward computation.
        to_update_classifier (bool): Whether to refit the classifier on the selected templates.
        init_fidx (int): Index of the initial feature to always include in templates.
        n_tmpls (int): Number of templates to select and return.
        n_tdata_minibatch (int): Number of data points per minibatch for gradient mutation.
        n_cands_minibatch (int): Number of candidate templates per minibatch.
        n_cands_targ (int): Target number of candidate templates to generate.
        min_features (int): Minimum number of features in each template.
        max_features (Optional[int]): Maximum number of features in each template.
        lmbda (float): Regularization parameter penalizing template size.
        bsz (int): Batch size for reward computation.
        make_opt_fn (Callable[[Iterable[th.Tensor]], th.optim.Optimizer]): Function to create an optimizer for template parameters.
        n_gradient_mutate_iters (int): Number of outer iterations for candidate mutation.
        n_gradient_steps_per_mutate_iter (int): Number of gradient steps per mutation iteration.
        plf (pl.Fabric): PyTorch Lightning Fabric object for device management.

    Returns:
        th.Tensor: Selected binary template matrix of shape (n_tmpls, n_features).
    """

    def _uniform_like(inputs: th.Tensor, low: float, high: float):
        return th.distributions.Uniform(low, high).sample(inputs.shape)

    generator = th.default_generator if generator is None else generator
    classifier.eval().to(device=plf.device)
    n_data: int = len(tdata)
    n_covs: int = tdata["xs"].shape[1]
    # use gradient descent to construct high quality templates
    ctmpls: th.Tensor | None = None
    pbar = tqdm.trange(
        n_gradient_mutate_iters, desc="gd-cands", leave=False, dynamic_ncols=True
    )
    for _itr in pbar:
        # draw a "tiny" batch of training instances
        _bidxs: th.Tensor = (
            th.multinomial(
                th.ones(n_data),
                num_samples=n_tdata_minibatch,
                replacement=False,
                generator=generator,
            )
            if n_tdata_minibatch < n_data
            else th.arange(n_data)
        )
        _bsz: int = len(_bidxs)
        _bdata: thd.TensorDict = tdata[_bidxs]
        _btxs: th.Tensor = _bdata["xs"]
        _btys: th.Tensor = _bdata["ys"]
        # each tiny batches are seeded with randomly mutated candidates
        # add some noise to input candidates, which also prevents nan in logit
        _bctmpls: th.Tensor = tafalib_makers_candidates.make_template_candidates(
            n_covs=n_covs,
            init_fidx=init_fidx,
            n_cands_targ=n_cands_minibatch,
            min_features=min_features,
            max_features=max_features,
            generator=generator,
        )
        _bctmpls = th.where(
            _bctmpls == 1,
            _bctmpls - _uniform_like(_bctmpls, 0.1, 0.5),
            _bctmpls + _uniform_like(_bctmpls, 0.1, 0.5),
        )
        # gd works with real, so transofrm indicator to logits
        _blctmpls: th.Tensor = th.logit(_bctmpls, eps=1e-6).requires_grad_(True)
        # make optimizer for current batch of data
        _bopt: th.optim.Optimizer = make_opt_fn([_blctmpls])
        _bpbar = tqdm.trange(
            n_gradient_steps_per_mutate_iter,
            desc="gdmutate-batch",
            dynamic_ncols=True,
            leave=False,
        )
        _bctmplidxs_set: set[int] = set()
        for _step in _bpbar:
            # (_bsz,  n_cands)
            _brwds_l: list[th.Tensor] = list()
            with th.autograd.graph.save_on_cpu():
                for _bbidxs in th.split(
                    th.cartesian_prod(th.arange(_bsz), th.arange(len(_bctmpls))), bsz
                ):
                    _bbctxs: th.Tensor = _btxs[_bbidxs[:, 0], :].to(device=plf.device)
                    _bblacts: th.Tensor = _blctmpls[_bbidxs[:, 1], :].to(
                        device=plf.device
                    )
                    _bbacts: th.Tensor = th.sigmoid(_bblacts)
                    _bbpyhats: th.Tensor = classifier.predict_proba(_bbctxs, _bbacts)
                    # (bsz, )
                    _bbcels: th.Tensor = th.nn.functional.nll_loss(
                        th.log(_bbpyhats),
                        _btys[_bbidxs[:, 0]].to(device=plf.device),
                        reduction="none",
                    )
                    _bbrwds: th.Tensor = -_bbcels - lmbda * th.sum(_bbacts, dim=1)
                    _brwds_l.append(_bbrwds)
            _brwds: th.Tensor = th.unflatten(
                th.cat(_brwds_l, dim=0), dim=0, sizes=(_bsz, n_cands_minibatch)
            )
            _brwds, _bupdtmpls_idxs = th.max(_brwds, dim=1)
            _bupdtmpls_idxs = _bupdtmpls_idxs.to(device="cpu")
            _blctmpls_entropy: th.Tensor = (
                th.distributions.Bernoulli(logits=_blctmpls[_bupdtmpls_idxs])
                .entropy()
                .mean()
            )
            _bloss: th.Tensor = -th.mean(_brwds)
            _bopt.zero_grad()
            _bloss.backward()
            _bopt.step()
            _bpbar.set_postfix(
                {"loss": _bloss.item(), "lctmpls_entropy": _blctmpls_entropy.item()}
            )
            plf.log_dict(
                {
                    f"gdvanilla/loss@itr{_itr}": _bloss.item(),
                    f"gdvanilla/lctmpls_entropy@itr{_itr}": _blctmpls_entropy.item(),
                },
                step=_step,
            )
            # keep track of which templates are updated
            _bctmplidxs_set.update(th.unique(_bupdtmpls_idxs).tolist())
        _blctmpls = _blctmpls.detach_().requires_grad_(False)
        # those never get touched by gradient descent are bad candidates
        # choose only those that have actually been touched by gradient descent
        _blctmpls = _blctmpls[list(_bctmplidxs_set)]
        # from logits back to feature masks
        _bctmpls = th.sigmoid(_blctmpls)
        # make it feauture masks
        _bctmpls = th.where(_bctmpls < 0.5, 0, 1).to(dtype=th.long, device="cpu")
        _bctmpls[:, init_fidx] = 1
        ctmpls = (
            th.unique(_bctmpls, dim=0)
            if ctmpls is None
            else th.unique(th.cat((ctmpls, _bctmpls), dim=0), dim=0)
        )
        assert ctmpls is not None
        pbar.set_postfix({"len": len(ctmpls)})
        plf.log_dict({"gdvanilla/len(ctmpls)": len(ctmpls)}, step=_itr)
        if len(ctmpls) > n_cands_targ:
            break
    assert ctmpls is not None
    # fill remaining holes with random templates
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = _fill_fcs_set_with_random_tmpls(
        fcs_set={tuple(_c.tolist()) for _c in ctmpls},
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=n_cands_targ,
        min_features=min_features,
        max_features=max_features,
        prv_featcounts=None,
        generator=generator,
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = len(fcs_l)
    ctmpls = th.zeros((n_cands, n_covs), dtype=th.long)
    for _i, _fc in enumerate(fcs_l):
        ctmpls[_i, _fc] = 1
    # compute rewards for each instance using each of the candidates
    tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
        tmpls=ctmpls,
        data=(
            tdata[
                th.multinomial(
                    th.ones((len(tdata),)), num_samples=max_tdata, generator=generator
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
    # choose a collection of templates from candidates
    tmpls, slctd_ms = make_templates_from_candidates(
        tpcomp=tpcomp,
        ctmpls=ctmpls,
        n_tmpls=n_tmpls,
        plf=plf,
        log_prefix="vanilla_mkgdtmpl",
    )
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(tmpls)
    return tmpls


# NOTE make tempaltes from candidates
def make_templates_from_candidates(
    tpcomp: thd.TensorDict,
    ctmpls: th.Tensor,
    n_tmpls: int,
    plf: pl.Fabric,
    log_prefix: Optional[str] = "mk_tmpl",
) -> tuple[th.Tensor, th.Tensor]:
    """make templates given a collection of candidates

    Args:
        tpcomp (thd.TensorDict): (n, ) precomputed results from using `tafalib.utils.precomp_rwds_for_tmpls`
        ctmpls (th.Tensor): (n_cands, n_covs) collection of candidates
        n_tmpls (int): number of templates
        plf (pl.Fabric): lightning fabric instance
        log_prefix (Optional[str], optional): prefix for logging. Defaults to "mk_tmpl".

    Returns::
        th.Tensor: (n_tmpls, n_covs) collection of identified templates
        th.Tensor: (n_cands, ) indicator vector over candidates, i.e. `tmpls = n_cands[slctd_ms]`
    """
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
def make_templates_direct_greedy_with_undo(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls: int,
    max_features: Optional[int],
    lmbda: float,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
    max_tdata: Optional[int] = None,
    tmpls_pt: Optional[th.Tensor] = None,
    eval_every_n_iter: int = 1,
    n_neighs: Optional[int] = None,
    vdata: Optional[thd.TensorDict] = None,
    vclassifier: Optional[mymodels.classifiers.SubsetFeatureClassifier] = None,
    metrics_func: Optional[thm.MetricCollection] = None,
    ckpt_p: Optional[str] = None,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """
    Direct greedy template optimization with undo capability.

    This function implements a sophisticated direct greedy algorithm for optimizing binary feature
    templates with the ability to modify existing templates (the "undo" mechanism). Unlike traditional
    greedy approaches that only add new templates, this algorithm can toggle features on/off in
    existing templates, allowing for more flexible optimization.

    The algorithm iteratively improves a collection of binary feature templates by:
    1. Starting with initialized templates (either from scratch or pre-trained)
    2. For each template, generating candidate modifications by toggling each feature
    3. Creating candidate template collections by combining modified templates with unchanged ones
    4. Evaluating the fitness of each candidate collection using classifier rewards
    5. Selecting the collection with the highest improvement in fitness
    6. Optionally validating performance on a separate validation set

    The "undo" capability allows the algorithm to remove previously selected features if doing so
    improves the overall template collection performance, making it more adaptive than standard
    greedy approaches.

    Args:
        tdata (thd.TensorDict): Training data containing 'xs' (features) and 'ys' (labels).
        classifier (mymodels.classifiers.SubsetFeatureClassifier): Classifier used to evaluate
            template performance and compute rewards.
        init_fidx (int): Index of the initial feature that must be included in all templates.
            This feature cannot be toggled off.
        n_tmpls (int): Target number of templates to optimize.
        max_features (Optional[int]): Maximum number of features per template. If None,
            uses the total number of features (n_covs).
        lmbda (float): Regularization parameter that penalizes templates with more features.
            Higher values encourage sparser templates.
        n_iter (int): Maximum number of optimization iterations to perform.
        bsz (int): Batch size for processing template evaluations.
        plf (pl.Fabric): PyTorch Lightning Fabric instance for device management and logging.
        max_tdata (Optional[int]): optional maximum training data to subsample if training data
          is too large and takes too long to complete. Defaults to None.
        tmpls_pt (Optional[th.Tensor]): Pre-trained templates to initialize from. If None,
            starts with blank templates. Shape: (n_tmpls, n_covs). Defaults to None.
        eval_every_n_iter (int): Frequency of validation evaluation (every N iterations).
            Defaults to 1.
        n_neighs (Optional[int]): Number of nearest neighbors to consider for validation cost
            estimation. If None, validation evaluation is skipped. Defaults to None.
        vdata (Optional[thd.TensorDict]): Validation data for periodic performance evaluation.
            If None, no validation is performed. Defaults to None.
        vclassifier (Optional[mymodels.classifiers.SubsetFeatureClassifier]): Classifier for
            validation evaluation. If None, uses the training classifier. Defaults to None.
        metrics_func (Optional[thm.MetricCollection]): Collection of metrics to compute during
            validation. If None, validation evaluation is skipped. Defaults to None.
        ckpt_p (Optional[str]): Directory path for saving checkpoints. If None, no checkpoints
            are saved. Defaults to None.
        generator (Optional[th.Generator]): Random number generator for reproducibility.
            If None, uses the default generator. Defaults to None.

    Returns:
        th.Tensor: Optimized binary template matrix of shape (n_tmpls, n_covs), where each
            row represents a template and each element is 0 or 1 indicating feature inclusion.

    Algorithm Details:
        - **Direct Optimization**: Generates candidate modifications on-the-fly rather than
          using pre-computed candidate pools.
        - **Feature Toggling**: For each template, considers adding or removing each feature
          (except the initial feature which is always included).
        - **Collection-based Evaluation**: Evaluates entire template collections rather than
          individual templates, accounting for redundancy and complementarity.
        - **Adaptive Termination**: Stops early if no improvements can be found and the
          target number of templates is reached.
        - **Progressive Building**: Gradually builds up to the target number of templates,
          allowing for incremental optimization.

    Example:
        ```python
        # Initialize with 5 templates, maximum 10 features each
        templates = make_templates_direct_greedy_with_undo(
            tdata=training_data,
            classifier=my_classifier,
            init_fidx=0,    # Always include feature 0
            n_tmpls=5,
            max_features=10,
            lmbda=0.1,      # Light regularization
            n_iter=100,
            bsz=32,
            plf=fabric,
            tmpls_pt=None,  # Start from scratch
            eval_every_n_iter=10,
            n_neighs=5,
            vdata=validation_data,
            vclassifier=None,
            metrics_func=metrics
        )
        ```

    Notes:
        - The function requires gradient computation to be disabled (@th.no_grad()).
        - Validation evaluation uses k-nearest neighbors cost estimation.
        - Progress is logged through the Lightning Fabric logging system.
        - Checkpoints save the current template state at each iteration if enabled.
    """
    generator = th.default_generator if generator is None else generator
    vclassifier = classifier if vclassifier is None else vclassifier
    if ckpt_p is not None:
        os.makedirs(ckpt_p, exist_ok=True)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # initialize templates
    tmpls: th.Tensor = (
        th.zeros((n_tmpls, n_covs), dtype=th.long) if tmpls_pt is None else tmpls_pt
    )
    tmpls[:, init_fidx] = 1
    if tmpls_pt is not None:
        assert n_tmpls == len(tmpls)
    # book keeping
    _nxt_blk_idx: int = 0 if tmpls_pt is None else n_tmpls - 1
    # direct greedy optimize
    pbar = tqdm.trange(n_iter, desc="direct-greedy", leave=False, dynamic_ncols=True)
    for _itr in pbar:
        _tdata: thd.TensorDict = (
            tdata
            if max_tdata is None or len(tdata) <= max_tdata
            else tdata[
                th.multinomial(
                    th.ones((len(tdata),)), num_samples=max_tdata, replacement=False
                )
            ]
        )
        # NOTE compute tpcomp for previous round of tmpls
        # (_ntmpls_prv, ncovs)
        _tmpls_prv: th.Tensor = tmpls[0 : _nxt_blk_idx + 1]
        # (ntdata, )
        _tpcomps_prv: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            tmpls=_tmpls_prv,
            data=_tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        # NOTE compute fitness of previous templates
        # (ntdata, )
        _rwds_prv: th.Tensor = th.max(_tpcomps_prv["rwds"], dim=1)[0]
        # () fitness_prv is a scaler
        _fitness_prv: th.Tensor = th.mean(_rwds_prv)
        # NOTE make all possible future templates without duplication
        _ntmpls: int = len(_tmpls_prv)
        _uctmpls_inds: set[frozenset[tuple[int, ...]]] = set()
        # for each template, generate "successor" collection of tmpls
        for _tidx in range(_ntmpls):
            # copy all templates except the one indexed by _tidx
            _base_inds: set[tuple[int, ...]] = {
                tuple(_t.tolist()) for _i, _t in enumerate(_tmpls_prv) if _i != _tidx
            }
            # given tidx-th template, for each possible location, toggle avaialability of one feature to tidx-th template
            for _pos in [_i for _i in range(n_covs) if _i != init_fidx]:
                _cinds: set[tuple[int, ...]] = copy.deepcopy(_base_inds)
                _new_ind: th.Tensor = _tmpls_prv[_tidx].clone()
                _new_ind[_pos] = 1 if _new_ind[_pos] == 0 else 0
                if th.sum(_new_ind) > max_features:
                    continue
                _cinds.add(tuple(_new_ind.tolist()))
                # filter out the case where the _new_ind is not something new to _base_inds
                if len(_cinds) == len(_base_inds):
                    continue
                # add the complete next potential template collection back to _tctmpls_inds
                _uctmpls_inds.add(frozenset(_cinds))
        # NOTE make list of potential future templates
        # a list of (_ncands, ) where each entry is (_ntmpls, )
        _ctmpls_l: list[th.Tensor] = [th.tensor([*_inds]) for _inds in _uctmpls_inds]
        # NOTE compute rewards
        # make unique feature combinations
        # keep track of all the information to recover the order
        _ctmpls_lens_l: list[int] = list(map(len, _ctmpls_l))
        _ctmpls_flt: th.Tensor
        _inv_idxs: th.Tensor
        _ctmpls_flt, _inv_idxs = th.unique(
            th.cat(_ctmpls_l, dim=0), dim=0, return_inverse=True
        )
        # list of (_ncands, ) each entry being a tensor of indices to _ctmpls_flt
        # the i-th entry of the list is the indices to which template is used to compose i-th candidate
        _inv_idxs_l: tuple[th.Tensor, ...] = th.split(_inv_idxs, _ctmpls_lens_l)
        # NOTE compute rewards
        # (ntdata, )
        _ctpcomps_flt: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            tmpls=_ctmpls_flt,
            data=_tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        # (ntdata, len(_ctmpls_flt))
        _crwds_flt: th.Tensor = _ctpcomps_flt["rwds"]
        # NOTE compute fitness
        # (_ncands, )
        _cfitness: th.Tensor = th.stack(
            [th.mean(th.max(_crwds_flt[:, _idxs], dim=1)[0]) for _idxs in _inv_idxs_l]
        )
        # NOTE compute improvements
        # (_ncands, )
        _improvements: th.Tensor = _cfitness - _fitness_prv
        _improvements = th.clamp(_improvements, min=0.0)
        # NOTE terminate loop if no imiprovement is made
        if not th.any(_improvements > 0.0) and _nxt_blk_idx + 1 >= n_tmpls:
            break
        # NOTE choose the one with the most improvement
        _slctd: int = int(th.argmax(_improvements, dim=0).item())
        tmpls[0 : _nxt_blk_idx + 1] = _ctmpls_l[_slctd]
        # NOTE if the blank template is modified, increment _nxt_blk_idx
        if th.all(th.sum(_ctmpls_l[_slctd], dim=1) != 1) and _nxt_blk_idx + 1 < n_tmpls:
            _nxt_blk_idx = _nxt_blk_idx + 1
        # NOTE log metrics
        _metrics_d: dict[str, float] = {
            "fitness": _cfitness[_slctd].item(),
            "improvement": _improvements[_slctd].item(),
        }
        pbar.set_postfix(_metrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(_metrics_d, "make-templates"), step=_itr
        )
        # NOTE keep track of validation set rollout performance
        if (
            n_neighs is not None
            and vdata is not None
            and vclassifier is not None
            and metrics_func is not None
            and _itr % eval_every_n_iter == 0
        ):
            # (ntdata, _ncands)
            _ctcels: th.Tensor = _ctpcomps_flt["cels"][:, _inv_idxs_l[_slctd]]
            _vmetrics_d: dict[str, float] = tafalib_utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib_functional.knn_cost_est(
                    x,
                    lmbda=lmbda,
                    txs=_tdata["xs"],
                    tcels=_ctcels,
                    tmpls=_ctmpls_l[_slctd],
                    n_neighs=n_neighs,
                    p=2,
                ),
                init_fidx=init_fidx,
                tmpls=_ctmpls_l[_slctd],
                lmbda=lmbda,
                metrics_func=metrics_func,
                plf=plf,
            )
            plf.log_dict(mylib.utils.add_prefix_to_dict(_vmetrics_d, "val"), step=_itr)
        if ckpt_p is not None:
            th.save(tmpls, os.path.join(ckpt_p, f"tmpls_itr{_itr}.pt"))
    pbar.close()
    return tmpls


# NOTE temperature function for direct gradient method
def simple_exponential_decay_temperature(
    itr: int,
    decay_every_n_iter=100,
    init_temperature=1.0,
    rate=9e-1,
    min_temperature: float = 1e-7,
) -> float:
    num_steps = itr // decay_every_n_iter
    temperature: float = init_temperature * (rate**num_steps)
    if temperature < min_temperature:
        temperature = min_temperature
    return temperature


# NOTE uniqueness penalty cost function
def collision_probability_penalty(ltmpls: th.Tensor) -> th.Tensor:
    """
    Penalize the probability that two rows will be identical in discrete space.

    Args:
        logits: (M, D) tensor of logits
        lambda_reg: penalty strength

    Mathematical intuition:
    - For binary variables, P(bit_i == bit_j) = p_i * p_j + (1-p_i) * (1-p_j)
    - For entire rows to be identical, ALL bits must match
    - P(row_i == row_j) = ∏_d P(bit_i[d] == bit_j[d])
    """
    P = th.sigmoid(ltmpls)  # Convert logits to probabilities [0,1]
    M, D = P.shape
    # Expand dimensions for broadcasting
    # P_i: (M, 1, D) - each row i broadcast across dimension 1
    # P_j: (1, M, D) - each row j broadcast across dimension 0
    P_i = P.unsqueeze(1)  # Shape: (M, 1, D)
    P_j = P.unsqueeze(0)  # Shape: (1, M, D)
    # For each pair (i,j) and each dimension d:
    # P(bit_i[d] == bit_j[d]) = P_i[d] * P_j[d] + (1-P_i[d]) * (1-P_j[d])
    same_bit_prob = P_i * P_j + (1 - P_i) * (1 - P_j)
    # Shape: (M, M, D)
    # P(row_i == row_j) = ∏_d P(bit_i[d] == bit_j[d])
    # In log space: log(∏) = ∑log, then exp back
    # But torch.prod works directly on small probabilities
    collision_prob = th.prod(same_bit_prob, dim=-1)  # Shape: (M, M)
    # Remove diagonal (self-comparisons) and avoid double-counting
    # Upper triangular mask excludes diagonal and lower triangle
    mask = th.triu(th.ones(M, M, device=ltmpls.device), diagonal=1)
    return th.sum(collision_prob * mask)


def expected_hamming_penalty(ltmpls: th.Tensor) -> th.Tensor:
    """
    Maximize expected Hamming distance between all pairs of rows.

    Mathematical intuition:
    - Hamming distance = number of differing bits
    - E[HD(row_i, row_j)] = ∑_d P(bit_i[d] ≠ bit_j[d])
    - P(bit_i[d] ≠ bit_j[d]) = p_i*(1-p_j) + (1-p_i)*p_j
    """
    P = th.sigmoid(ltmpls)
    M, D = P.shape
    P_i = P.unsqueeze(1)  # (M, 1, D)
    P_j = P.unsqueeze(0)  # (1, M, D)
    # P(bit differs) = P_i * (1-P_j) + (1-P_i) * P_j
    # This is the probability that bit i is 1 and bit j is 0, OR
    # bit i is 0 and bit j is 1
    diff_bit_prob = P_i * (1 - P_j) + (1 - P_i) * P_j
    # Shape: (M, M, D)
    # Expected Hamming distance = sum over all dimensions
    expected_hamming = th.sum(diff_bit_prob, dim=-1)  # Shape: (M, M)
    # We want to MAXIMIZE Hamming distance, so negative penalty
    mask = th.triu(th.ones(M, M, device=ltmpls.device), diagonal=1)
    # Negative because we want to maximize distance (minimize negative distance)
    return -th.sum(expected_hamming * mask)


def simple_cycle(tdata: thd.TensorDict, block_size: int) -> Generator[tuple[int, ...]]:
    """
    Generate a simple block-based feature cycling pattern for cyclic greedy optimization.

    This function creates a generator that infinitely cycles through blocks of feature indices,
    designed to be used as the feature_opt_cycler parameter in make_templates_cyclic_greedy_with_undo.
    It divides all available features into equally-sized blocks and yields each block in sequence,
    repeating the pattern indefinitely.

    The cycling pattern helps control the computational complexity of the cyclic greedy algorithm
    by limiting the number of features considered at each iteration, while ensuring all features
    are eventually considered through the cycling process.

    Args:
        tdata (thd.TensorDict): Training data containing feature information. Used to determine
            the total number of features (n_covs) from the shape of tdata["xs"].
        block_size (int): Number of features to include in each cycle block. Features are
            grouped into blocks of this size. The last block may contain fewer features
            if the total number of features is not evenly divisible by block_size.

    Yields:
        tuple[int, ...]: A tuple containing feature indices for the current cycle block.
            The generator cycles through all blocks infinitely, restarting from the first
            block after the last block is yielded.

    Example:
        ```python
        # For data with 10 features and block_size=3:
        # Will create blocks: (0,1,2), (3,4,5), (6,7,8), (9,)
        # Then cycle: (0,1,2) -> (3,4,5) -> (6,7,8) -> (9,) -> (0,1,2) -> ...

        cycler = simple_cycle(tdata, block_size=3)

        # Use with cyclic greedy optimization
        templates = make_templates_cyclic_greedy_with_undo(
            tdata=tdata,
            classifier=classifier,
            init_fidx=0,
            n_tmpls=5,
            max_features=10,
            lmbda=0.1,
            feature_opt_cycler=cycler,  # Use the block-based cycler
            n_iter=100,
            bsz=32,
            plf=fabric
        )
        ```

    Notes:
        - The generator is infinite - it will continue cycling through blocks indefinitely.
        - Block size should be chosen based on computational constraints and desired granularity:
        * Smaller blocks: More focused optimization, potentially slower convergence
        * Larger blocks: Less focused optimization, faster per-iteration processing
        - Features are grouped sequentially (0,1,2...), but you can create custom cycling
        patterns by implementing alternative generators.
        - The last block may be smaller than block_size if n_covs is not evenly divisible.
    """
    n_covs: int = tdata["xs"].shape[1]
    cycles: list[tuple[int, ...]] = [
        tuple(_curr_pos.tolist())
        for _curr_pos in th.split(th.arange(0, n_covs, dtype=th.long), block_size)
    ]
    for _curr_pos in itrtls.cycle(cycles):
        yield _curr_pos


@th.no_grad()
def make_templates_cyclic_greedy_with_undo(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls: int,
    max_features: Optional[int],
    lmbda: float,
    feature_opt_cycler: Generator[tuple[int, ...]],
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
    max_tdata: Optional[int] = None,
    tmpls_pt: Optional[th.Tensor] = None,
    eval_every_n_iter: int = 1,
    n_neighs: Optional[int] = None,
    vdata: Optional[thd.TensorDict] = None,
    vclassifier: Optional[mymodels.classifiers.SubsetFeatureClassifier] = None,
    metrics_func: Optional[thm.MetricCollection] = None,
    ckpt_p: Optional[str] = None,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """
    Cyclic greedy template optimization with undo capability.

    This function implements a cyclic greedy algorithm for optimizing binary feature
    templates with the ability to modify existing templates (the "undo" mechanism). Unlike
    the direct greedy approach that considers all features simultaneously, this algorithm
    cycles through predetermined subsets of features, making it more computationally
    efficient and allowing for structured exploration of the feature space.

    The algorithm iteratively improves a collection of binary feature templates by:
    1. Starting with initialized templates (either from scratch or pre-trained)
    2. Cycling through feature subsets provided by the feature_opt_cycler
    3. For each template, generating candidate modifications by toggling features only
    within the current cycle subset
    4. Creating candidate template collections by combining modified templates with unchanged ones
    5. Evaluating the fitness of each candidate collection using classifier rewards
    6. Selecting the collection with the highest improvement in fitness
    7. Optionally validating performance on a separate validation set

    The "undo" capability allows the algorithm to remove previously selected features if doing so
    improves the overall template collection performance, making it more adaptive than standard
    greedy approaches. The cyclic nature provides structured optimization that can be more
    efficient than exhaustive feature consideration.

    Args:
        tdata (thd.TensorDict): Training data containing 'xs' (features) and 'ys' (labels).
        classifier (mymodels.classifiers.SubsetFeatureClassifier): Classifier used to evaluate
            template performance and compute rewards.
        init_fidx (int): Index of the initial feature that must be included in all templates.
            This feature cannot be toggled off.
        n_tmpls (int): Target number of templates to optimize.
        max_features (Optional[int]): Maximum number of features per template. If None,
            uses the total number of features (n_covs).
        lmbda (float): Regularization parameter that penalizes templates with more features.
            Higher values encourage sparser templates.
        feature_opt_cycler (Generator[tuple[int, ...]]): Generator that yields tuples of
            feature indices to consider at each iteration. This controls which features
            are eligible for toggling in each optimization cycle.
        n_iter (int): Maximum number of optimization iterations to perform.
        bsz (int): Batch size for processing template evaluations.
        plf (pl.Fabric): PyTorch Lightning Fabric instance for device management and logging.
        max_tdata (Optional[int]): Optional maximum training data to subsample if training data
            is too large and takes too long to complete. Defaults to None.
        tmpls_pt (Optional[th.Tensor]): Pre-trained templates to initialize from. If None,
            starts with blank templates. Shape: (n_tmpls, n_covs). Defaults to None.
        eval_every_n_iter (int): Frequency of validation evaluation (every N iterations).
            Defaults to 1.
        n_neighs (Optional[int]): Number of nearest neighbors to consider for validation cost
            estimation. If None, validation evaluation is skipped. Defaults to None.
        vdata (Optional[thd.TensorDict]): Validation data for periodic performance evaluation.
            If None, no validation is performed. Defaults to None.
        vclassifier (Optional[mymodels.classifiers.SubsetFeatureClassifier]): Classifier for
            validation evaluation. If None, uses the training classifier. Defaults to None.
        metrics_func (Optional[thm.MetricCollection]): Collection of metrics to compute during
            validation. If None, validation evaluation is skipped. Defaults to None.
        ckpt_p (Optional[str]): Directory path for saving checkpoints. If None, no checkpoints
            are saved. Defaults to None.
        generator (Optional[th.Generator]): Random number generator for reproducibility.
            If None, uses the default generator. Defaults to None.

    Returns:
        th.Tensor: Optimized binary template matrix of shape (n_tmpls, n_covs), where each
            row represents a template and each element is 0 or 1 indicating feature inclusion.

    Algorithm Details:
        - **Cyclic Optimization**: Only considers features from the current cycle subset rather
        than all features simultaneously, reducing computational complexity.
        - **Feature Toggling**: For each template, considers adding or removing each feature
        within the current cycle (except the initial feature which is always included).
        - **Collection-based Evaluation**: Evaluates entire template collections rather than
        individual templates, accounting for redundancy and complementarity.
        - **Adaptive Termination**: Stops early if no improvements can be found and the
        target number of templates is reached.
        - **Progressive Building**: Gradually builds up to the target number of templates,
        allowing for incremental optimization.

    Example:
        ```python
        # Create a cycler that considers features in blocks of 5
        cycler = simple_cycle(tdata, block_size=5)

        # Initialize with 3 templates, maximum 8 features each
        templates = make_templates_cyclic_greedy_with_undo(
            tdata=training_data,
            classifier=my_classifier,
            init_fidx=0,              # Always include feature 0
            n_tmpls=3,
            max_features=8,
            lmbda=0.1,               # Light regularization
            feature_opt_cycler=cycler, # Cycle through feature blocks
            n_iter=50,
            bsz=32,
            plf=fabric,
            tmpls_pt=None,           # Start from scratch
            eval_every_n_iter=5,
            n_neighs=3,
            vdata=validation_data,
            vclassifier=None,
            metrics_func=metrics
        )
        ```

    Notes:
        - The function requires gradient computation to be disabled (@th.no_grad()).
        - The feature_opt_cycler determines the optimization schedule - use simple_cycle()
        for block-based cycling or implement custom cycling strategies.
        - Validation evaluation uses k-nearest neighbors cost estimation.
        - Progress is logged through the Lightning Fabric logging system.
        - Checkpoints save the current template state at each iteration if enabled.
        - The cyclic approach can be significantly faster than direct greedy when the
        feature space is large.
    """
    generator = th.default_generator if generator is None else generator
    vclassifier = classifier if vclassifier is None else vclassifier
    if ckpt_p is not None:
        os.makedirs(ckpt_p, exist_ok=True)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # initialize templates
    tmpls: th.Tensor = (
        th.zeros((n_tmpls, n_covs), dtype=th.long) if tmpls_pt is None else tmpls_pt
    )
    tmpls[:, init_fidx] = 1
    if tmpls_pt is not None:
        assert n_tmpls == len(tmpls)
    # book keeping
    _nxt_blk_idx: int = 0 if tmpls_pt is None else n_tmpls - 1
    # cyclic greedy optimize
    pbar = tqdm.trange(n_iter, desc="cyclic-greedy", leave=False, dynamic_ncols=True)
    for _itr, _curr_cycle in zip(pbar, feature_opt_cycler):
        _tdata: thd.TensorDict = (
            tdata
            if max_tdata is None or len(tdata) <= max_tdata
            else tdata[
                th.multinomial(
                    th.ones((len(tdata),)), num_samples=max_tdata, replacement=False
                )
            ]
        )
        # NOTE compute tpcomp for previous round of tmpls
        # (_ntmpls_prv, ncovs)
        _tmpls_prv: th.Tensor = tmpls[0 : _nxt_blk_idx + 1]
        # (ntdata, )
        _tpcomps_prv: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            tmpls=_tmpls_prv,
            data=_tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        # NOTE compute fitness of previous templates
        # (ntdata, )
        _rwds_prv: th.Tensor = th.max(_tpcomps_prv["rwds"], dim=1)[0]
        # () fitness_prv is a scaler
        _fitness_prv: th.Tensor = th.mean(_rwds_prv)
        # NOTE make all possible future templates without duplication
        _ntmpls: int = len(_tmpls_prv)
        _uctmpls_inds: set[frozenset[tuple[int, ...]]] = set()
        # for each template, generate "successor" collection of tmpls
        for _tidx in range(_ntmpls):
            # copy all templates except the one indexed by _tidx
            _base_inds: set[tuple[int, ...]] = {
                tuple(_t.tolist()) for _i, _t in enumerate(_tmpls_prv) if _i != _tidx
            }
            # given tidx-th template, for each possible location, toggle avaialability of one feature to tidx-th template
            for _pos in [_i for _i in _curr_cycle if _i != init_fidx]:
                _cinds: set[tuple[int, ...]] = copy.deepcopy(_base_inds)
                _new_ind: th.Tensor = _tmpls_prv[_tidx].clone()
                _new_ind[_pos] = 1 if _new_ind[_pos] == 0 else 0
                if th.sum(_new_ind) > max_features:
                    continue
                _cinds.add(tuple(_new_ind.tolist()))
                # filter out the case where the _new_ind is not something new to _base_inds
                if len(_cinds) == len(_base_inds):
                    continue
                # add the complete next potential template collection back to _tctmpls_inds
                _uctmpls_inds.add(frozenset(_cinds))
        # NOTE make list of potential future templates
        # a list of (_ncands, ) where each entry is (_ntmpls, )
        _ctmpls_l: list[th.Tensor] = [th.tensor([*_inds]) for _inds in _uctmpls_inds]
        # NOTE compute rewards
        # make unique feature combinations
        # keep track of all the information to recover the order
        _ctmpls_lens_l: list[int] = list(map(len, _ctmpls_l))
        _ctmpls_flt: th.Tensor
        _inv_idxs: th.Tensor
        _ctmpls_flt, _inv_idxs = th.unique(
            th.cat(_ctmpls_l, dim=0), dim=0, return_inverse=True
        )
        # list of (_ncands, ) each entry being a tensor of indices to _ctmpls_flt
        # the i-th entry of the list is the indices to which template is used to compose i-th candidate
        _inv_idxs_l: tuple[th.Tensor, ...] = th.split(_inv_idxs, _ctmpls_lens_l)
        # NOTE compute rewards
        # (ntdata, )
        _ctpcomps_flt: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
            tmpls=_ctmpls_flt,
            data=_tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        # (ntdata, len(_ctmpls_flt))
        _crwds_flt: th.Tensor = _ctpcomps_flt["rwds"]
        # NOTE compute fitness
        # (_ncands, )
        _cfitness: th.Tensor = th.stack(
            [th.mean(th.max(_crwds_flt[:, _idxs], dim=1)[0]) for _idxs in _inv_idxs_l]
        )
        # NOTE compute improvements
        # (_ncands, )
        _improvements: th.Tensor = _cfitness - _fitness_prv
        _improvements = th.clamp(_improvements, min=0.0)
        # NOTE terminate loop if no imiprovement is made
        if not th.any(_improvements > 0.0) and _nxt_blk_idx + 1 >= n_tmpls:
            break
        # NOTE choose the one with the most improvement
        _slctd: int = int(th.argmax(_improvements, dim=0).item())
        tmpls[0 : _nxt_blk_idx + 1] = _ctmpls_l[_slctd]
        # NOTE if the blank template is modified, increment _nxt_blk_idx
        if th.all(th.sum(_ctmpls_l[_slctd], dim=1) != 1) and _nxt_blk_idx + 1 < n_tmpls:
            _nxt_blk_idx = _nxt_blk_idx + 1
        # NOTE log metrics
        _metrics_d: dict[str, float] = {
            "fitness": _cfitness[_slctd].item(),
            "improvement": _improvements[_slctd].item(),
        }
        pbar.set_postfix(_metrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(_metrics_d, "make-templates"), step=_itr
        )
        # NOTE keep track of validation set rollout performance
        if (
            n_neighs is not None
            and vdata is not None
            and vclassifier is not None
            and metrics_func is not None
            and _itr % eval_every_n_iter == 0
        ):
            # (ntdata, _ncands)
            _ctcels: th.Tensor = _ctpcomps_flt["cels"][:, _inv_idxs_l[_slctd]]
            _vmetrics_d: dict[str, float] = tafalib_utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib_functional.knn_cost_est(
                    x,
                    lmbda=lmbda,
                    txs=_tdata["xs"],
                    tcels=_ctcels,
                    tmpls=_ctmpls_l[_slctd],
                    n_neighs=n_neighs,
                    p=2,
                ),
                init_fidx=init_fidx,
                tmpls=_ctmpls_l[_slctd],
                lmbda=lmbda,
                metrics_func=metrics_func,
                plf=plf,
            )
            plf.log_dict(mylib.utils.add_prefix_to_dict(_vmetrics_d, "val"), step=_itr)
        if ckpt_p is not None:
            th.save(tmpls, os.path.join(ckpt_p, f"tmpls_itr{_itr}.pt"))
    pbar.close()
    return tmpls


def make_templates_direct_minibatch_gradient(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    tmpls_pt: Optional[th.Tensor],
    init_fidx: int,
    n_tmpls: int,
    max_features: Optional[int],
    lmbda: float,
    make_opt_fn: Callable[[Iterable[th.Tensor]], th.optim.Optimizer],
    n_iter: int,
    bsz: int,
    bsz_compute_oracle: int,
    alpha_unq: float,
    get_temperature_fn: Callable[[int], float],
    compute_uniqueness_penalty_fn: Callable[[th.Tensor], th.Tensor],
    plf: pl.Fabric,
    n_neighs: int,
    vdata: Optional[thd.TensorDict],
    vclassifier: Optional[mymodels.classifiers.SubsetFeatureClassifier],
    metrics_func: thm.MetricCollection,
    eval_every_n_iter: int,
    ckpt_p: Optional[str] = None,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    # initialize
    assert isinstance(
        classifier, mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier
    )
    generator = th.default_generator if generator is None else generator
    vclassifier = classifier if vclassifier is None else vclassifier
    assert isinstance(
        vclassifier, mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier
    )
    if ckpt_p is not None:
        os.makedirs(ckpt_p, exist_ok=True)
    classifier = classifier.eval().to(device=plf.device)
    vclassifier = vclassifier.eval().to(device=plf.device)
    tloader = th_data.DataLoader(
        tdata, batch_size=bsz, collate_fn=lambda x: x  # type:ignore
    )
    # record shapes
    n_covs: int = tdata["xs"].shape[1]
    max_features = n_covs if max_features is None else max_features
    # initialize templates in logit space
    # init_fidx-th column shall not be updated by gradient descent
    # (n_tmpls, n_covs)
    ltmpls_o: th.Tensor = (
        th.nn.init.orthogonal_(
            th.empty(
                (n_tmpls, n_covs - 1),
                dtype=th.float32,
                device=plf.device,
            )
        )
        if tmpls_pt is None
        else torch.distributions.utils.probs_to_logits(
            th.clamp(tmpls_pt, 0.3, 0.7), is_binary=True
        )[:, [_i for _i in range(n_covs) if _i != init_fidx]].to(device=plf.device)
    )
    ltmpls_o = ltmpls_o.requires_grad_(True)
    EPS_TRUE_PROB: th.Tensor = torch.distributions.utils.probs_to_logits(
        th.tensor(9.9999999999e-1, dtype=th.float32, device=plf.device), is_binary=True
    )
    get_ltmpls_fn: Callable[[], th.Tensor] = lambda: th.cat(  # noqa: E731
        (
            ltmpls_o[:, :init_fidx],
            EPS_TRUE_PROB[None, None].expand(n_tmpls, 1),
            ltmpls_o[:, init_fidx:],
        ),
        dim=1,
    )
    # make optimizer
    opt: th.optim.Optimizer = make_opt_fn([ltmpls_o])
    opt.zero_grad()
    # direct greedy optimize
    if tmpls_pt is not None:
        assert n_tmpls == len(ltmpls_o)
    pbar = tqdm.trange(n_iter, desc="direct-gradient", leave=False, dynamic_ncols=True)
    _step: int = 0
    for _itr in pbar:
        _temperature: float = get_temperature_fn(_itr)
        for _btdata in tloader:
            _btdata: thd.TensorDict
            _bsz: int = len(_btdata)
            # (_btdata, n_covs)
            _btxs: th.Tensor = _btdata["xs"]
            # (_btdata, )
            _btys: th.Tensor = _btdata["ys"]
            #
            _ltmpls: th.Tensor = get_ltmpls_fn()
            # NOTE draw a random sample from loogit template
            _btmpls: th.Tensor = th.distributions.RelaxedBernoulli(
                temperature=th.tensor(_temperature, device=plf.device), logits=_ltmpls
            ).rsample()

            # NOTE compute cross entropy loss for each instance and template combination
            _brwds_l: list[th.Tensor]

            def _minibatch_compute_oracle_rwd(_bbidxs: th.Tensor) -> th.Tensor:
                # (_bsz, n_covs)
                _bbctxs: th.Tensor = _btxs[_bbidxs[:, 0], :].to(device=plf.device)
                _bbacts: th.Tensor = _btmpls[_bbidxs[:, 1], :].to(device=plf.device)
                # (_bsz, n_labels)
                _bbpyhats: th.Tensor = classifier.predict_proba(_bbctxs, _bbacts)
                # (_bsz, )
                _bbcels: th.Tensor = th.nn.functional.cross_entropy(
                    _bbpyhats,
                    _btys[_bbidxs[:, 0]].to(device=plf.device),
                    reduction="none",
                )
                _bbrwds: th.Tensor = -_bbcels - lmbda * th.sum(_bbacts, dim=1)
                return _bbrwds

            with th.autograd.graph.save_on_cpu():
                _brwds_l = [
                    _minibatch_compute_oracle_rwd(_bbidxs)
                    for _bbidxs in th.split(
                        th.cartesian_prod(th.arange(_bsz), th.arange(n_tmpls)),
                        bsz_compute_oracle,
                    )
                ]
            # (_bsz, n_tmpls)
            _brwds: th.Tensor = th.unflatten(
                th.cat(_brwds_l, dim=0), dim=0, sizes=(_bsz, n_tmpls)
            )
            # NOTE compute costs
            # (_bsz, n_tmpls)
            _bweights: th.Tensor = th.distributions.RelaxedOneHotCategorical(
                temperature=th.tensor(_temperature, device=plf.device), logits=_brwds
            ).rsample()
            # (_bsz, )
            _bcosts: th.Tensor = th.mean(-_brwds * _bweights, dim=1)
            _bunq_loss: th.Tensor = compute_uniqueness_penalty_fn(_ltmpls)
            # compute penalties
            # TODO encourage sparsity, encourage pairwise independence
            # compute loss
            _bloss: th.Tensor = th.mean(_bcosts) + alpha_unq * _bunq_loss
            # back prop but not update
            _bloss.backward()
            # NOTE log metrics
            _metrics_d: dict[str, float] = {
                "temperature": _temperature,
                "cost": th.mean(_bcosts).item(),
                "uniqueness_loss": _bunq_loss.item(),
                "loss": _bloss.item(),
                "gd-update": 0,
            }
            pbar.set_postfix({"loss": _bloss.item()})
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(_metrics_d, "minibatch-gradient"),
                step=_step,
            )
            _step = _step + 1
        opt.step()
        opt.zero_grad()
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(
                {"gd-update": 1},
                "minibatch-gradient",
            ),
            step=_step,
        )
        # NOTE keep track of validation set rollout performance
        if vdata is not None and _itr % eval_every_n_iter == 0:
            _tmpls: th.Tensor = th.where(
                th.sigmoid(get_ltmpls_fn()) < 0.5, 0.0, 1.0
            ).to(device="cpu")
            _tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
                _tmpls, data=tdata, classifier=classifier, lmbda=lmbda, bsz=bsz, plf=plf
            )
            _vmetrics_d: dict[str, float] = tafalib_utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib_functional.knn_cost_est(
                    x,
                    lmbda=lmbda,
                    txs=tdata["xs"],
                    tcels=_tpcomp["cels"],
                    tmpls=_tmpls,
                    n_neighs=n_neighs,
                    p=2,
                ),
                init_fidx=init_fidx,
                tmpls=_tmpls,
                lmbda=lmbda,
                metrics_func=metrics_func,
                plf=plf,
            )
            plf.log_dict(mylib.utils.add_prefix_to_dict(_vmetrics_d, "val"), step=_itr)
        if ckpt_p is not None:
            th.save(get_ltmpls_fn(), os.path.join(ckpt_p, f"ltmpls_itr{_itr}.pt"))
    pbar.close()
    tmpls: th.Tensor = th.where(th.sigmoid(get_ltmpls_fn()) < 0.5, 0.0, 1.0).to(
        dtype=th.int64, device="cpu"
    )
    return tmpls


# NOTE internal
@th.no_grad()
def _ident_init_fidx_single(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    max_features: Optional[int],
    n_masks: int,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
    generator: Optional[th.Generator] = None,
) -> tuple[int, th.Tensor]:
    generator = th.default_generator if generator is None else generator
    n_covs: int = tdata["xs"].shape[1]
    classifier.eval().to(device=plf.device)
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(
            tafalib_makers_candidates.make_feature_masks(
                n_covs=n_covs,
                n_masks=len(tdata),
                min_features=1,
                max_features=max_features,
                generator=generator,
            )
        )
    fms: th.Tensor = tafalib_makers_candidates.make_feature_masks(
        n_covs=n_covs,
        n_masks=n_masks,
        min_features=1,
        max_features=max_features,
        generator=generator,
    )
    best_mask_cost: th.Tensor = th.inf * th.ones((n_covs,))
    n_masks = min(n_masks, len(fms))
    for _i in tqdm.trange(
        n_masks, desc="ident_init_fidx_single", leave=False, dynamic_ncols=True
    ):
        # (n_covs, )
        _bfm: th.Tensor = fms[_i]
        # (bsz, )
        _btidxs: th.Tensor = th.randint(0, len(tdata), (bsz,), generator=generator)
        # (bsz, n_covs)
        _btdata: thd.TensorDict = tdata[_btidxs]
        _bfms: th.Tensor = _bfm[None, :].expand(bsz, -1)
        # (bsz, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(
            _btdata["xs"].to(device=plf.device), _bfms.to(device=plf.device)
        ).to(device="cpu")
        _blyhats: th.Tensor = torch.distributions.utils.probs_to_logits(_bpyhats)
        # ()
        _bcosts: th.Tensor = th.nn.functional.cross_entropy(
            _blyhats, _btdata["ys"]
        ) + lmbda * th.sum(_bfm)
        best_mask_cost = th.minimum(
            best_mask_cost, th.where(_bfm == 1, _bcosts, th.inf)
        )
    return int(th.argmin(best_mask_cost).item()), best_mask_cost


def _update_template_candidates(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    init_fidx: int,
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
    generator: Optional[th.Generator],
) -> th.Tensor:
    """update template candidates

    Args:
        ctmpls (th.Tensor): (n_cands, n_covs) collection of candidates used in previous round
        slctd_ms (th.Tensor): (n_cands, ) indicator vector over `ctmpls` s.t. collection of tempaltes found in previous round `tmpls = ctmpls[slctd_ms]`
        init_fidx (int): initial feature index
        n_cands_targ (int): target number of candidates
        min_features (int): minimum number of features included in each candidate
        max_features (Optional[int]): optional maximum number of features included in each candidates; set to `n_covs` if `None`.
        use_feature_importance_sampling (bool): use feautre frequencies from previous round identified templates to sample candidates for next round of search.

    Returns:
        th.Tensor: (n_cands, n_covs) a new collection of candidates.
    """
    generator = th.default_generator if generator is None else generator
    fcs_set: set[tuple[int, ...]] = _mutate_tmpls(
        tmpls_prv=ctmpls[slctd_ms],
        init_fidx=init_fidx,
        n_cands_targ=n_cands_targ,
        min_features=min_features,
        generator=generator,
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
        generator=generator,
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = min(n_cands_targ, len(fcs_l))
    ctmpls_new: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    for _i, _fc in enumerate(fcs_l):
        ctmpls_new[_i, _fc] = 1
    return ctmpls_new


def _update_template_candidates_fix_rounds(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    init_fidx: Optional[int | _PostHocIdentifyInitialFeatureFunc],
    n_cands_init: int,
    n_cands_mutate: int | None,
    n_cands_targ: int | None,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
    generator: Optional[th.Generator],
) -> th.Tensor:
    generator = th.default_generator if generator is None else generator
    tmpls_prv: th.Tensor = ctmpls[slctd_ms]
    n_cands_mutate = len(tmpls_prv) if n_cands_mutate is None else n_cands_mutate
    fcs_set: set[tuple[int, ...]] = _mutate_tmpls(
        tmpls_prv=tmpls_prv,
        init_fidx=init_fidx,
        n_cands_targ=min(n_cands_mutate + len(tmpls_prv), n_cands_init),
        min_features=min_features,
        generator=generator,
    )
    n_covs: int = ctmpls.shape[1]
    n_cands_targ = (
        5 * len(tmpls_prv) + len(fcs_set) if n_cands_targ is None else n_cands_targ
    )
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = _fill_fcs_set_with_random_tmpls(
        fcs_set=fcs_set,
        n_covs=n_covs,
        init_fidx=init_fidx,
        n_cands_targ=min(n_cands_targ, n_cands_init),
        min_features=min_features,
        max_features=max_features,
        prv_featcounts=(
            th.sum(ctmpls[slctd_ms], dim=0) if use_feature_importance_sampling else None
        ),
        generator=generator,
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = len(fcs_l)
    ctmpls_new: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    for _i, _fc in enumerate(fcs_l):
        ctmpls_new[_i, _fc] = 1
    return ctmpls_new


def _mutate_tmpls(
    tmpls_prv: th.Tensor,
    init_fidx: Optional[int | _PostHocIdentifyInitialFeatureFunc],
    n_cands_targ: int,
    min_features: int,
    generator: Optional[th.Generator],
) -> set[tuple[int, ...]]:
    generator = th.default_generator if generator is None else generator
    # new candidate pool set
    fcs_set: set[tuple[int, ...]] = {
        tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls_prv
    }
    # previous template pool and exclude those that has no feature to mutate from
    tmpls_prv = tmpls_prv[th.sum(tmpls_prv, dim=1) - min_features > 0]
    if len(tmpls_prv) == 0:
        return fcs_set
    # mutate templates
    pbar = tqdm.tqdm(
        itrtls.count(), desc="mutate tmpl_prv", dynamic_ncols=True, leave=False
    )
    for _c in pbar:
        if _c >= n_cands_targ or len(fcs_set) >= n_cands_targ:
            break
        # randomly choose an existing template to mutate from
        _tmpl_prv: th.Tensor = tmpls_prv[
            int(th.randint(0, len(tmpls_prv), (), generator=generator).item())
        ]
        # make copy of selected template
        _tmpl: th.Tensor = _tmpl_prv.clone()
        # always remove one feature from currently chosen templates
        _nfeats_mut: int = 1
        # indices to previously selected feature
        _fidxs: th.Tensor = th.argwhere(_tmpl == 1).flatten()
        # prevent init_fidx from being mutated
        if init_fidx is not None and isinstance(init_fidx, int):
            _fidxs = _fidxs[_fidxs != init_fidx]
        # set mutated feature to be zero
        _tmpl[
            _fidxs[
                th.multinomial(
                    th.ones_like(_fidxs, dtype=th.float64),
                    _nfeats_mut,
                    generator=generator,
                )
            ]
        ] = 0
        # add _tmpl to candidate pool
        _fc: tuple[int, ...] = tuple(th.argwhere(_tmpl == 1).flatten().tolist())
        if _fc not in fcs_set:
            fcs_set.add(_fc)
            pbar.set_postfix({"len": len(fcs_set)})
    pbar.close()
    return fcs_set


def _fill_fcs_set_with_random_tmpls(
    fcs_set: set[tuple[int, ...]],
    n_covs: int,
    init_fidx: Optional[int | _PostHocIdentifyInitialFeatureFunc],
    n_cands_targ: int,
    min_features: int,
    max_features: Optional[int],
    prv_featcounts: Optional[th.Tensor],
    generator: Optional[th.Generator],
) -> list[set[tuple[int, ...]]]:
    generator = th.default_generator if generator is None else generator
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
        return fcs_sets_by_bins
    # subtract existing fcs from the bins
    bincnt_fcs = bincnt_fcs - th.as_tensor(
        list(map(len, fcs_sets_by_bins)), dtype=th.long
    )
    # sample number of fcs to add to existing feature combinations
    ps: th.Tensor = th.ones_like(bincnt_fcs, dtype=th.float32)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(
            ps, n_cands_targ - len(fcs_set), replacement=True, generator=generator
        ),
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
                generator=generator,
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
            _fc_l: list[int] = th.multinomial(
                _ps, num_samples=_nfeats, generator=generator
            ).tolist()
            # make sure initial feature is in fcomb
            if (
                init_fidx is not None
                and isinstance(init_fidx, int)
                and init_fidx not in _fc_l
            ):
                _fc_l.append(init_fidx)
                _fc_l = _fc_l[1:]
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc = tuple(_fc_l)
            if _fc not in _fcs_set:
                _fcs_set.add(_fc)
    return fcs_sets_by_bins
