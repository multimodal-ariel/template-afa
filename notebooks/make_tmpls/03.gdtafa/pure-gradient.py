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
        _bctmpls: th.Tensor = tafalib.makers.candidates.make_template_candidates(
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
            _bctmpls
            if ctmpls is None
            else th.unique(th.cat((ctmpls, _bctmpls), dim=0), dim=0)
        )
        if len(ctmpls) > n_cands_targ:
            break
    assert ctmpls is not None
    tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
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
    tmpls, slctd_ms = tafalib.makers.templates.make_templates_from_candidates(
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
n_tdata_minibatch: int = 1024
n_cands_minibatch: int = 1024
bsz: int = 2048000
lr: float = 1
n_gradient_mutate_iters: int = 10
n_gradient_steps_per_mutate_iter: int = 50


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
output_dir: str = os.path.join("outputs", "run", data_name, "gdtafa-hybrid")
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
tmpls = make_vanilla_gradient_descent_templates(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    to_update_classifier=False,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls_targ,
    n_tdata_minibatch=n_tdata_minibatch,
    n_cands_minibatch=n_cands_minibatch,
    n_cands_targ=n_cands_targ,
    min_features=min_features_targ,
    max_features=max_features_targ,
    lmbda=lmbda,
    bsz=bsz,
    make_opt_fn=lambda p: th.optim.AdamW(p, lr=lr),
    n_gradient_mutate_iters=n_gradient_mutate_iters,
    n_gradient_steps_per_mutate_iter=n_gradient_steps_per_mutate_iter,
    plf=plf,
)

# %%
