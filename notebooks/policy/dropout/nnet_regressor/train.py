# %%
from __future__ import annotations

import os
from typing import Optional, TypedDict

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
# mktmpl_run_dir: str = "experiments/make_template/outputs/big5_cnnet/20250314_112541/14"
mktmpl_run_dir: str = "experiments/make_template/outputs/big5_cnnet/20250318_144121/5"


# %%
class _TrainState(TypedDict):
    nnet: th.nn.Module
    opt: th.optim.Optimizer
    fit_itr: int
    opt_step: int


@th.no_grad()
def compile_selector_dataset(
    tdata: thd.TensorDict, tpcomp: thd.TensorDict
) -> thd.TensorDict:
    # (n_data, n_covs)
    xs: th.Tensor = tdata["xs"]
    # (n_data, )
    ys: th.Tensor = tdata["ys"]
    # (n_data, n_tmpls, n_labels)
    pyhats: th.Tensor = tpcomp["pyhats"]
    # (n_data, n_tmpls)
    cels: th.Tensor = tpcomp["cels"]
    rwds: th.Tensor = tpcomp["rwds"]
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
        }
    ).auto_batch_size_(1)
    return stdata


@th.no_grad()
def _make_fit_bsinps(
    bstdata: thd.TensorDict, init_fidx: int, lmbda: float, tmpls: th.Tensor
) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    tmpls = tmpls.to(device=bxs.device)
    bfms: th.Tensor = tmpls[btmplidxs]
    n_covs: int = bxs.shape[1]
    # randomly drop features
    boms: th.Tensor = th.randint(0, 2, (bsz, n_covs), device=bxs.device)
    boms = th.clamp(bfms - boms, 0.0, 1.0)
    boms[:, init_fidx] = 1
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs * boms, boms), dim=1)
    return bsinps


def _fit_iter_nnet_regressor(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    lmbda: float,
    plf: pl.Fabric,
) -> dict[str, float]:
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # (bsz, 2 * n_covs)
    bsinps = _make_fit_bsinps(
        bstdata=bstdata, init_fidx=init_fidx, lmbda=lmbda, tmpls=tmpls
    )
    # (bsz, n_tmpls)
    bsouts: th.Tensor = nnet(bsinps)
    # compute selector loss
    bslosses: th.Tensor = th.nn.functional.mse_loss(
        bsouts, bstdata["cels"], reduction="none"
    )
    bsloss: th.Tensor = th.mean(bslosses)
    # update selector parameter
    opt.zero_grad()
    bsloss.backward()
    opt.step()
    # track metrics
    bmetrics_d: dict[str, float] = {
        "bsloss": bsloss.item(),
    }
    tstate["opt_step"] = tstate["opt_step"] + 1
    return bmetrics_d


def fit_nnet_regressor(
    tstate: _TrainState,
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
        bmetrics_d: dict[str, float] = _fit_iter_nnet_regressor(
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
stdata: thd.TensorDict = compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)

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
tstate = _TrainState(
    nnet=nnet,
    opt=opt,
    fit_itr=0,
    opt_step=0,
)

# %%
fit_nnet_regressor(
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
    cost_est=lambda x: tafalib.functional.multi_output_nnet_cost_est(
        x, nnet=nnet, lmbda=mktmpl_cfg.lmbda, tmpls=tmpls, device=plf.device
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
