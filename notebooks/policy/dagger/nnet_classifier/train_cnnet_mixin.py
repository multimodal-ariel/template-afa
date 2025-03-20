# %%
from __future__ import annotations

import copy
import os
from typing import Optional

import _daggerlib
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import pandas as pd
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import torchrl.data as thrl_data
import tqdm.auto as tqdm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = mylib.utils.get_project_root_dir()

# %%
# mktmpl_run_dir: str = "experiments/make_template/outputs/grid/20250305_145621/2"
# mktmpl_run_dir: str = "experiments/make_template/outputs/cube/20250305_143844/4"
mktmpl_run_dir: str = "experiments/make_template/outputs/big5_cnnet/20250314_112541/14"


# %%
def warmup_fit_nnet_selector(
    tstate: _daggerlib._TrainState,
    stdata: thd.TensorDict,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
    ckpt_p: Optional[str] = None,
    save_ckpt_every_n_iter: int = 1,
):
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _itr in pbar:
        bstdata: thd.TensorDict = stdata[th.randint(0, len(stdata), (bsz,))].to(
            device=plf.device
        )
        # track metrics
        bmetrics_d: dict[str, float] = _daggerlib._warmup_fit_iter_nnet_selector(
            tstate=tstate,
            bstdata=bstdata,
            init_fidx=init_fidx,
            lmbda=lmbda,
            tmpls=tmpls,
            plf=plf,
        )
        pbar.set_postfix(bmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(bmetrics_d, "train_warmup"),
            step=tstate["opt_step"],
        )
        if ckpt_p is not None and (
            (_itr % save_ckpt_every_n_iter) == 0 or (_itr + 1) == n_iter
        ):
            plf.save(os.path.join(ckpt_p, f"warmup_itr{_itr}.pt"), tstate)
    pbar.close()
    return tstate


def dagger_fit_nnet_selector(
    tstate: _daggerlib._TrainState,
    stdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    to_reset_nnet_after_dagger_rollout: bool,
    n_iter: int,
    n_dagger_rollout: int,
    n_opt_step_per_iter: int,
    bsz: int,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
    vdata: Optional[thd.TensorDict] = None,
    vclassifier: Optional[mymodels.classifiers.SubsetFeatureClassifier] = None,
    eval_every_n_iter: int = 1,
    ckpt_p: Optional[str] = None,
    save_ckpt_every_n_iter: int = 1,
):
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    warmup_data: thd.TensorDict = tstate["rplbuf"][:].clone()
    for _itr in pbar:
        n_rollout: int = min(n_dagger_rollout, len(stdata))
        bstdata: thd.TensorDict = stdata[
            th.multinomial(th.ones((len(stdata),)), n_rollout, replacement=False)
        ]
        # dataset aggregation (dagger)
        _daggerlib._update_dagger_replay_buffer_(
            tstate=tstate,
            bstdata=bstdata,
            classifier=classifier,
            init_fidx=init_fidx,
            lmbda=lmbda,
            tmpls=tmpls,
            plf=plf,
        )
        # if we want to train from scratch, then we need to re-initialize the weight
        if to_reset_nnet_after_dagger_rollout:
            for _module in tstate["nnet"].modules():
                if hasattr(_module, "reset_parameters"):
                    _module.reset_parameters()
        # update nnet
        for _s in tqdm.trange(n_opt_step_per_iter, dynamic_ncols=True, leave=False):
            btmetrics_d: dict[str, float]
            if _s % 2 == 0:
                # use buffer from dagger
                btmetrics_d = _daggerlib._dagger_fit_iter_nnet_selector(
                    tstate=tstate, bsz=bsz, plf=plf
                )
            else:
                # use warmup data
                tstate["nnet"].train().to(device=plf.device)
                _bidxs: th.Tensor = th.multinomial(
                    th.ones((len(warmup_data),)),
                    min(bsz, len(warmup_data)),
                    replacement=False,
                )
                _bdata: thd.TensorDict = warmup_data[_bidxs].to(plf.device)
                # (bsz, n_tmpls)
                _bsouts: th.Tensor = tstate["nnet"](_bdata["inps"])
                # compute selector loss
                _bslosses: th.Tensor = th.nn.functional.cross_entropy(
                    _bsouts, _bdata["targs"], reduction="none"
                )
                _bsloss: th.Tensor = th.mean(_bslosses)
                # update selector parameter
                tstate["opt"].zero_grad()
                _bsloss.backward()
                tstate["opt"].step()
                # track metrics
                btmetrics_d: dict[str, float] = {
                    "bsloss": _bsloss.item(),
                }
                tstate["opt_step"] = tstate["opt_step"] + 1
            # track metrics
            pbar.set_postfix(btmetrics_d)
            btmetrics_d["fit_itr"] = tstate["fit_itr"]
            btmetrics_d["opt_step"] = tstate["opt_step"]
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(btmetrics_d, "train_dagger"),
                step=tstate["opt_step"],
            )
        # save ckpt if needed
        if ckpt_p is not None and (
            (_itr % save_ckpt_every_n_iter) == 0 or (_itr + 1) == n_iter
        ):
            plf.save(os.path.join(ckpt_p, f"dagger_itr{_itr}.pt"), tstate)
        # rollout with validation set to test performance if needed
        if vdata is not None and (
            _itr % eval_every_n_iter == 0 or (_itr + 1) == n_iter
        ):
            vclassifier = classifier if vclassifier is None else vclassifier
            vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
                    x, nnet=tstate["nnet"], device=plf.device
                ),
                init_fidx=init_fidx,
                tmpls=tmpls,
                metrics_func=metrics_func,
                plf=plf,
            )
            vmetrics_d["fit_itr"] = tstate["fit_itr"]
            vmetrics_d["opt_step"] = tstate["opt_step"]
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(vmetrics_d, "train_dagger"),
                step=tstate["fit_itr"],
            )
        # update fit counter
        tstate["fit_itr"] = tstate["fit_itr"] + 1
    pbar.close()
    return tstate


# %%
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tdata_shuffle_idxs.pt")
)
tmpls: th.Tensor = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tmpls.pt"), weights_only=False
)
tpcomp: thd.TensorDict = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tpcomp.pt"), weights_only=False
)
mktmpl_cfg = OmegaConf.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, ".hydra", "config.yaml")
)

# %%
# make dataset
_tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
_tdata, vdata, tstdata = hd.utils.call(mktmpl_cfg.data)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
# split training data into two for classifier and afa
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
# make classifier
# NOTE cnnet
tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
    mktmpl_cfg.tclassifier,
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
)
if os.path.exists(
    os.path.join(mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt")
):
    tclassifier.load_state_dict(
        th.load(
            os.path.join(
                mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
            )
        )
    )
# NOTE xgboost
# tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
#     cfg.tclassifier, xs_train=extdata["xs"].numpy(), ys_train=extdata["ys"].numpy()
# )
if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
    tclassifier_p = os.path.join(
        mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
    )
    if os.path.exists(tclassifier_p):
        tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
    else:
        tclassifier.fit_(tmpls)

# %%
vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
if mktmpl_cfg.vclassifier is not None:
    vclassifier = hd.utils.instantiate(
        mktmpl_cfg.vclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )

# %%
stdata: thd.TensorDict = _daggerlib.compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", mktmpl_cfg.data.name, "cnnet_mixin")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

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
# sanity check if saved classifier
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.knn_cost_est(
        x,
        lmbda=mktmpl_cfg.lmbda,
        txs=tdata["xs"],
        tcels=tpcomp["cels"],
        tmpls=tmpls,
        n_neighs=mktmpl_cfg.n_neighs,
        p=2,
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
nnet = mymodels.nn.make_fcn(
    in_features=2 * tdata["xs"].shape[1],
    out_features=len(tmpls),
    layer_specs=[
        (tdata["xs"].shape[1], th.nn.BatchNorm1d, th.nn.ReLU, 0.5),
        (tdata["xs"].shape[1], th.nn.BatchNorm1d, th.nn.ReLU, 0.5),
    ],
)

# %%
opt = th.optim.Adam(nnet.parameters(), lr=1e-4)

# %%
tstate = _daggerlib._TrainState(
    rplbuf=thrl_data.ReplayBuffer(
        storage=thrl_data.ListStorage(max_size=100_000_000),
        sampler=thrl_data.SamplerWithoutReplacement(),
    ),
    nnet=nnet,
    opt=opt,
    fit_itr=0,
    opt_step=0,
)

# %%
warmup_fit_nnet_selector(
    tstate=tstate,
    stdata=stdata,
    init_fidx=mktmpl_cfg.init_fidx,
    lmbda=mktmpl_cfg.lmbda,
    tmpls=tmpls,
    n_iter=1_000,
    bsz=1024,
    plf=plf,
)

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.knn_cost_est(
        x,
        lmbda=mktmpl_cfg.lmbda,
        txs=tdata["xs"],
        tcels=tpcomp["cels"],
        tmpls=tmpls,
        n_neighs=mktmpl_cfg.n_neighs,
        p=2,
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
        x, nnet=nnet, device=plf.device
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
# use dagger to finetune cost estimator
dagger_fit_nnet_selector(
    tstate=tstate,
    stdata=stdata,
    classifier=tclassifier,
    init_fidx=mktmpl_cfg.init_fidx,
    lmbda=mktmpl_cfg.lmbda,
    tmpls=tmpls,
    to_reset_nnet_after_dagger_rollout=True,
    n_iter=50,
    n_dagger_rollout=len(tdata),
    n_opt_step_per_iter=25_000,
    bsz=1024,
    metrics_func=metrics_func,
    plf=plf,
    vdata=vdata,
    vclassifier=vclassifier,
    eval_every_n_iter=1,
    ckpt_p=None,
)

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.knn_cost_est(
        x,
        lmbda=mktmpl_cfg.lmbda,
        txs=tdata["xs"],
        tcels=tpcomp["cels"],
        tmpls=tmpls,
        n_neighs=mktmpl_cfg.n_neighs,
        p=2,
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
        x, nnet=nnet, device=plf.device
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
