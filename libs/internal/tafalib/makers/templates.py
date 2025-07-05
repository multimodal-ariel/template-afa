from __future__ import annotations

import itertools as itrtls
import math
from typing import Callable, Iterable, Optional

import lightning as pl
import mylib
import mymodels
import scipy.spatial as sp_spatial
import tensordict as thd
import torch as th
import torch.distributions.utils
import tqdm.auto as tqdm

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


# NOTE make templates from scratch
@th.no_grad()
def make_templates_vanilla(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: int,
    n_tmpls: int,
    n_cands: int,
    min_features: int,
    max_features: Optional[int],
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
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
    classifier.to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # NOTE init. candidate templates
    ctmpls: th.Tensor = tafalib_makers_candidates.make_template_candidates(
        n_covs=n_covs,
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
            tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
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
) -> th.Tensor:
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
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
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
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
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
                    )
                )
            ctmpls = th.cat(
                [
                    make_templates_vanilla(
                        tdata=tdata[
                            th.multinomial(
                                th.ones((len(tdata),)), num_samples=minibatch_size
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
                                    th.ones((len(tdata),)), num_samples=minibatch_size
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
                tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
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
            # initialize candidate pool
            ctmpls = tafalib_makers_candidates.make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
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
            )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(ctmpls)
        _tdata = (
            tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
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

    classifier.eval().to(device=plf.device)
    n_data: int = len(tdata)
    n_covs: int = tdata["xs"].shape[1]
    # use gradient descent to construct high quality templates
    ctmpls: th.Tensor | None = None
    for _itr in tqdm.trange(
        n_gradient_mutate_iters, desc="gd-cands", leave=False, dynamic_ncols=True
    ):
        # draw a "tiny" batch of training instances
        _bidxs: th.Tensor = (
            th.multinomial(
                th.ones(n_data), num_samples=n_tdata_minibatch, replacement=False
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
        for _ in _bpbar:
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
            _bloss: th.Tensor = -th.mean(th.max(_brwds, dim=1)[0])
            _bopt.zero_grad()
            _bloss.backward()
            _bopt.step()
            _bpbar.set_postfix({"loss": _bloss.item()})
        _blctmpls = _blctmpls.detach_().requires_grad_(False)
        _bctmpls = th.sigmoid(_blctmpls)
        _bctmpls = th.where(_bctmpls < 0.5, 0, 1).to(dtype=th.long, device="cpu")
        _bctmpls[:, init_fidx] = 1
        ctmpls = (
            th.unique(_bctmpls, dim=0)
            if ctmpls is None
            else th.unique(th.cat((ctmpls, _bctmpls), dim=0), dim=0)
        )
        assert ctmpls is not None
        if len(ctmpls) > n_cands_targ:
            break
    assert ctmpls is not None
    tpcomp: thd.TensorDict = tafalib_utils.precomp_rwds_for_tmpls(
        tmpls=ctmpls,
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
) -> tuple[int, th.Tensor]:
    n_covs: int = tdata["xs"].shape[1]
    classifier.eval().to(device=plf.device)
    if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        classifier.fit_(
            tafalib_makers_candidates.make_feature_masks(
                n_covs=n_covs,
                n_masks=len(tdata),
                min_features=1,
                max_features=max_features,
            )
        )
    fms: th.Tensor = tafalib_makers_candidates.make_feature_masks(
        n_covs=n_covs, n_masks=n_masks, min_features=1, max_features=max_features
    )
    best_mask_cost: th.Tensor = th.inf * th.ones((n_covs,))
    n_masks = min(n_masks, len(fms))
    for _i in tqdm.trange(
        n_masks, desc="ident_init_fidx_single", leave=False, dynamic_ncols=True
    ):
        # (n_covs, )
        _bfm: th.Tensor = fms[_i]
        # (bsz, )
        _btidxs: th.Tensor = th.randint(0, len(tdata), (bsz,))
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


def _update_template_candidates_fix_rounds(
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    init_fidx: int,
    n_cands_init: int,
    n_cands_mutate: int | None,
    n_cands_targ: int | None,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
) -> th.Tensor:
    tmpls_prv: th.Tensor = ctmpls[slctd_ms]
    n_cands_mutate = len(tmpls_prv) if n_cands_mutate is None else n_cands_mutate
    fcs_set: set[tuple[int, ...]] = _mutate_tmpls(
        tmpls_prv=tmpls_prv,
        init_fidx=init_fidx,
        n_cands_targ=min(n_cands_mutate + len(tmpls_prv), n_cands_init),
        min_features=min_features,
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
            pbar.set_postfix({"len": len(fcs_set)})
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
