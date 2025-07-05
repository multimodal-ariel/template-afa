# %%
from __future__ import annotations

import os
from typing import Callable, Iterable, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets
import mylib
import mymodels
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
# NOTE big5
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
run_p: str = (
    "experiments/pretrain/nnet_subset_feature_classifier/outputs/big5/20250312_224514"
)
run_cfg = OmegaConf.load(
    os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
)
tclassifier = (
    mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
        nnet=hd.utils.instantiate(
            run_cfg.nnet,
            in_features=n_covs * 2,
            out_features=n_labels,
        ),
        xs_train=tdata["xs"].numpy(),
        ys_train=tdata["ys"].numpy(),
        fit_kwargs=hd.utils.instantiate(run_cfg.nnet_fit_cfg),
        state_dict_p=os.path.join(
            mylib.utils.get_project_root_dir(), run_p, "classifier.pt"
        ),
    )
)
tclassifier.fit_kwargs["n_iter"] = 1000
tclassifier.fit_kwargs["bsz"] = 8192
vclassifier = tclassifier
max_tdata: Optional[int] = None
init_fidx: int = 35
n_tmpls_targ: int = 128
n_cands_targ: int = 10_000
n_cands_mutate: int = 256
lmbda: float = 0.075
min_features_targ: int = 1
max_features_targ: Optional[int] = None
min_features_init: int = 10
n_rounds: int = 3
feature_decrement: int = 2
use_feature_importance_sampling: bool = True
bsz: int = 409600
lr: float = 1e-2
n_gradient_mutate_iters: int = 200
n_gradient_steps_per_mutate_iter: int = 70000


# %%
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
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "gdtafa")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

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
@th.enable_grad()
def _mutate_candidates_with_gradient_descent(
    tdata: thd.TensorDict,
    tmpls_prv: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier,
    init_fidx: int,
    n_cands_targ: int,
    lmbda: float,
    bsz: int,
    make_opt_fn: Callable[[Iterable[th.Tensor]], th.optim.Optimizer],
    n_gradient_mutate_iters: int,
    n_gradient_steps_per_mutate_iter: int,
    plf: pl.Fabric,
):
    def _uniform_like(inputs: th.Tensor, low: float, high: float):
        return th.distributions.Uniform(low, high).sample(inputs.shape)

    classifier.eval().to(device=plf.device)
    n_data: int = len(tdata)
    n_cands: int = len(tmpls_prv)
    outs: th.Tensor = tmpls_prv.clone().detach_()
    for _itr in tqdm.trange(
        n_gradient_mutate_iters, desc="gd-mutate", leave=False, dynamic_ncols=True
    ):
        # draw a "tiny" batch of training instances
        _bidxs: th.Tensor = (
            th.multinomial(th.ones(n_data), num_samples=bsz, replacement=False)
            if bsz < n_data
            else th.arange(n_data)
        )
        _bsz: int = len(_bidxs)
        _bdata: thd.TensorDict = tdata[_bidxs]
        _btxs: th.Tensor = _bdata["xs"]
        _btys: th.Tensor = _bdata["ys"]
        # each tiny batches are seeded with randomly mutated candidates
        # add some noise to input candidates, which also prevents nan in logit
        _bctmpls = th.where(
            tmpls_prv == 1,
            tmpls_prv - _uniform_like(tmpls_prv, 0.2, 0.5),
            tmpls_prv + _uniform_like(tmpls_prv, 0.2, 0.5),
        )
        # gd works with real, so transofrm indicator to logits
        _blctmpls: th.Tensor = th.logit(_bctmpls, eps=1e-6).requires_grad_(True)
        # make optimizer for current batch of data
        _bopt: th.optim.Optimizer = make_opt_fn([_blctmpls])
        for _ in tqdm.trange(
            n_gradient_steps_per_mutate_iter,
            desc="gdmutate-batch",
            dynamic_ncols=True,
            leave=False,
        ):
            # (_bsz,  n_cands)
            _brwds_l: list[th.Tensor] = list()
            with th.autograd.graph.save_on_cpu():
                for _bbidxs in th.split(
                    th.cartesian_prod(th.arange(_bsz), th.arange(n_cands)), _bsz
                ):
                    _bbctxs: th.Tensor = _btxs[_bbidxs[:, 0], :].to(device=plf.device)
                    _bblacts: th.Tensor = _blctmpls[_bbidxs[:, 1], :].to(
                        device=plf.device
                    )
                    _bbacts: th.Tensor = th.sigmoid(_bblacts)
                    _bbpyhats: th.Tensor = classifier.predict_proba(_bbctxs, _bbacts)
                    # (_bsz, )
                    _bbcels: th.Tensor = th.nn.functional.nll_loss(
                        th.log(_bbpyhats),
                        _btys[_bbidxs[:, 0]].to(device=plf.device),
                        reduction="none",
                    )
                    _bbrwds: th.Tensor = -_bbcels - lmbda * th.sum(_bbacts, dim=1)
                    _brwds_l.append(_bbrwds)
            _brwds: th.Tensor = th.unflatten(
                th.cat(_brwds_l, dim=0), dim=0, sizes=(_bsz, n_cands)
            )
            _bloss: th.Tensor = -th.mean(th.max(_brwds, dim=1)[0])
            _bopt.zero_grad()
            _bloss.backward()
            _bopt.step()
        _blctmpls = _blctmpls.detach_().requires_grad_(False)
        _bctmpls = th.sigmoid(_blctmpls)
        _bctmpls = th.where(_bctmpls < 0.5, 0, 1).to(dtype=th.long, device="cpu")
        _bctmpls[:, init_fidx] = 1
        outs = th.unique(th.cat((outs, _bctmpls), dim=0), dim=0)
        if len(outs) > n_cands + n_cands_targ:
            _idxs: list[int] = list(range(n_cands))
            _idxs.extend(
                (
                    n_cands
                    + th.multinomial(
                        th.ones(len(outs) - n_cands),
                        num_samples=n_cands_targ,
                        replacement=False,
                    )
                )
                .sort()[0]
                .tolist()
            )
            outs = outs[_idxs]
            break
    return outs


def _update_template_candidates_fix_rounds_with_gradient_descent(
    tdata: thd.TensorDict,
    ctmpls: th.Tensor,
    slctd_ms: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier,
    init_fidx: int,
    n_cands_init: int,
    n_cands_mutate: int | None,
    n_cands_targ: int | None,
    min_features: int,
    max_features: Optional[int],
    use_feature_importance_sampling: bool,
    lmbda: float,
    bsz: int,
    make_opt_fn: Callable[[Iterable[th.Tensor]], th.optim.Optimizer],
    n_gradient_mutate_iters: int,
    n_gradient_steps_per_mutate_iter: int,
    plf: pl.Fabric,
) -> th.Tensor:
    tmpls_prv: th.Tensor = ctmpls[slctd_ms]
    n_cands_mutate = len(tmpls_prv) if n_cands_mutate is None else n_cands_mutate
    ctmpls_gnew: th.Tensor = _mutate_candidates_with_gradient_descent(
        tdata,
        tmpls_prv=tmpls_prv,
        classifier=classifier,
        init_fidx=init_fidx,
        n_cands_targ=2 * len(tmpls_prv) if n_cands_targ is None else n_cands_targ,
        lmbda=lmbda,
        bsz=bsz,
        make_opt_fn=make_opt_fn,
        n_gradient_mutate_iters=n_gradient_mutate_iters,
        n_gradient_steps_per_mutate_iter=n_gradient_steps_per_mutate_iter,
        plf=plf,
    )
    fcs_set: set[tuple[int, ...]] = tafalib.makers.templates._mutate_tmpls(
        tmpls_prv=tmpls_prv,
        init_fidx=init_fidx,
        n_cands_targ=min(n_cands_mutate + len(tmpls_prv), n_cands_init),
        min_features=min_features,
    )
    fcs_set.update(
        {tuple(th.argwhere(_c == 1).flatten().tolist()) for _c in ctmpls_gnew}
    )
    n_covs: int = ctmpls.shape[1]
    n_cands_targ = (
        2 * len(tmpls_prv) + len(fcs_set) if n_cands_targ is None else n_cands_targ
    )
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = (
        tafalib.makers.templates._fill_fcs_set_with_random_tmpls(
            fcs_set=fcs_set,
            n_covs=n_covs,
            init_fidx=init_fidx,
            n_cands_targ=min(n_cands_targ, n_cands_init),
            min_features=min_features,
            max_features=max_features,
            prv_featcounts=(
                th.sum(ctmpls[slctd_ms], dim=0)
                if use_feature_importance_sampling
                else None
            ),
        )
    )
    # from fcomb to act
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    n_cands: int = len(fcs_l)
    ctmpls_new: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
    for _i, _fc in enumerate(fcs_l):
        ctmpls_new[_i, _fc] = 1
    return ctmpls_new


@th.no_grad()
def make_templates_fix_rounds_with_gradient_descent(
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
    make_opt_fn: Callable[[Iterable[th.Tensor]], th.optim.Optimizer],
    n_gradient_mutate_iters: int,
    n_gradient_steps_per_mutate_iter: int,
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
    assert isinstance(
        classifier, mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier
    )
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
            ctmpls = tafalib.makers.candidates.make_template_candidates(
                n_covs=n_covs,
                init_fidx=init_fidx,
                n_cands_targ=n_cands_init,
                min_features=min_features,
                max_features=max_features,
            )
        else:
            # update candidate pool from existing templates
            ctmpls = _update_template_candidates_fix_rounds_with_gradient_descent(
                tdata=tdata,
                ctmpls=ctmpls,
                slctd_ms=slctd_ms,
                classifier=classifier,
                init_fidx=init_fidx,
                n_cands_init=n_cands_init,
                n_cands_mutate=n_cands_mutate,
                n_cands_targ=n_cands_targ,
                min_features=min_features,
                max_features=max_features,
                use_feature_importance_sampling=use_feature_importance_sampling,
                lmbda=lmbda,
                bsz=bsz,
                make_opt_fn=make_opt_fn,
                n_gradient_mutate_iters=n_gradient_mutate_iters,
                n_gradient_steps_per_mutate_iter=n_gradient_steps_per_mutate_iter,
                plf=plf,
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
            log_prefix=f"gdmutate_mktmpl{_i}",
        )
        if (
            isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
            and to_update_classifier
        ):
            classifier.fit_(tmpls)
    assert tmpls is not None
    return tmpls


# %%
tmpls = make_templates_fix_rounds_with_gradient_descent(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    to_update_classifier=False,
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
    make_opt_fn=lambda p: th.optim.SGD(p, lr=lr),
    n_gradient_mutate_iters=n_gradient_mutate_iters,
    n_gradient_steps_per_mutate_iter=n_gradient_steps_per_mutate_iter,
    plf=plf,
)

# %%
