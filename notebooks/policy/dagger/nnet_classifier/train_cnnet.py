# %%
from __future__ import annotations

import os
from typing import Callable, Optional, TypedDict

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import torchrl.data as thrl_data
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
mktmpl_run_dir: str = "experiments/make_template/outputs/big5_cnnet/20250314_112541/14"


# %%
class _TrainState(TypedDict):
    rplbuf: thrl_data.ReplayBuffer
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
def _make_warmup_fit_bsinps_bstargs(
    bstdata: thd.TensorDict, init_fidx: int, lmbda: float, tmpls: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
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
    # (bsz, n_tmpls, n_covs)
    bfms_avail: th.Tensor = th.clamp(tmpls[None, :, :] - boms[:, None, :], 0.0, 1.0)
    # (bsz, n_tmpls)
    bcels: th.Tensor = bstdata["cels"].to(device=bxs.device)
    bcosts: th.Tensor = bcels + lmbda * th.sum(bfms_avail, dim=2)
    # (bsz, )
    bstargs: th.Tensor = th.argmin(bcosts, dim=1)
    return bsinps, bstargs


def _warmup_fit_iter_nnet_selector(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    lmbda: float,
    plf: pl.Fabric,
) -> dict[str, float]:
    rplbuf: thrl_data.ReplayBuffer = tstate["rplbuf"]
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # (bsz, 2 * n_covs) (bsz, )
    bsinps, bstargs = _make_warmup_fit_bsinps_bstargs(
        bstdata=bstdata, init_fidx=init_fidx, lmbda=lmbda, tmpls=tmpls
    )
    rplbuf.extend(
        thd.make_tensordict(
            {"inps": bsinps.to(device="cpu"), "targs": bstargs.to(device="cpu")}
        ).auto_batch_size_(1)
    )
    # (bsz, n_tmpls)
    bsouts: th.Tensor = nnet(bsinps)
    # compute selector loss
    bslosses: th.Tensor = th.nn.functional.cross_entropy(
        bsouts, bstargs, reduction="none"
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


def warmup_fit_nnet_selector(
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
        bmetrics_d: dict[str, float] = _warmup_fit_iter_nnet_selector(
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


@th.no_grad()
def _make_dagger_fit_bsinps_bstargs(
    bstdata: thd.TensorDict,
    bfobsds_l: list[list[int]],
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
) -> tuple[list[th.Tensor], list[th.Tensor]]:
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bcels: th.Tensor = bstdata["cels"]
    tmpls = tmpls.to(device=bxs.device)
    # make new masks
    boms_l: list[th.Tensor] = list()
    for _bidx, _fobsd_l in enumerate(bfobsds_l):
        assert _fobsd_l[0] == init_fidx
        _oms_l: list[th.Tensor] = list()
        for _i in range(len(_fobsd_l)):
            _om: th.Tensor = th.zeros_like(bxs[_bidx])
            _om[_fobsd_l[: _i + 1]] = 1
            _oms_l.append(_om)
        boms_l.append(th.stack(_oms_l))
    # make new selector inputs
    bsinps_l: list[th.Tensor] = list()
    bstargs_l: list[th.Tensor] = list()
    for _bidx, _om in enumerate(boms_l):
        _xs: th.Tensor = bxs[_bidx][None, :].expand(len(_om), -1)
        _sinps: th.Tensor = th.cat((_xs * _om, _om), dim=1)
        _fms_avail: th.Tensor = th.clamp(tmpls[None, :, :] - _om[:, None, :], 0.0, 1.0)
        _cels: th.Tensor = bcels[_bidx][None, :]
        _costs: th.Tensor = _cels + lmbda * th.sum(_fms_avail, dim=2)
        _stargs: th.Tensor = th.argmin(_costs, dim=1)
        bsinps_l.append(_sinps)
        bstargs_l.append(_stargs)
    return bsinps_l, bstargs_l


@th.no_grad()
def _update_dagger_replay_buffer_(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    plf: pl.Fabric,
):
    rplbuf: thrl_data.ReplayBuffer = tstate["rplbuf"]
    nnet: th.nn.Module = tstate["nnet"].eval().to(device=plf.device)
    _, _, _, bfobsds_l = tafalib.utils.predict(
        data=bstdata,
        classifier=classifier,
        cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
            x, nnet=nnet, device=plf.device
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    nnet.train()
    # shape of list (bsz, ) (bsz, )
    # shape of 1st tensor (len(boms[_bi]), 2 * n_covs)
    # shape of 2nd tensor (len(boms[_bi]), )
    bsinps_l, bstargs_l = _make_dagger_fit_bsinps_bstargs(
        bstdata=bstdata,
        bfobsds_l=bfobsds_l,
        init_fidx=init_fidx,
        lmbda=lmbda,
        tmpls=tmpls,
    )
    # (sum(map(len, bsinps_l)), 2 * n_covs)
    bsinps: th.Tensor = th.cat(bsinps_l, dim=0).to(device="cpu")
    # (sum(map(len, bstargs_l)), )
    bstargs: th.Tensor = th.cat(bstargs_l, dim=0).to(device="cpu")
    assert len(bsinps) == len(bstargs)
    # (sum(map(len, bsinps_l)))
    bdata = thd.make_tensordict({"inps": bsinps, "targs": bstargs}).auto_batch_size_(1)
    # add bdata to replay buffer
    rplbuf.extend(bdata)


def _dagger_fit_iter_nnet_selector(
    tstate: _TrainState, bsz: int, plf: pl.Fabric
) -> dict[str, float]:
    rplbuf: thrl_data.ReplayBuffer = tstate["rplbuf"]
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # sample experiences from replay buffer
    bdata = rplbuf.sample(batch_size=bsz)
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = bdata["inps"].to(device=plf.device)
    # (bsz, )
    bstargs: th.Tensor = bdata["targs"].to(device=plf.device)
    # (bsz, n_tmpls)
    bsouts: th.Tensor = nnet(bsinps)
    # compute selector loss
    # (bsz, )
    bslosses: th.Tensor = th.nn.functional.cross_entropy(
        bsouts, bstargs, reduction="none"
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


def dagger_fit_nnet_selector(
    tstate: _TrainState,
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
    for _itr in pbar:
        n_rollout: int = min(n_dagger_rollout, len(stdata))
        bstdata: thd.TensorDict = stdata[th.randint(0, len(stdata), (n_rollout,))].to(
            device=plf.device
        )
        # dataset aggregation (dagger)
        _update_dagger_replay_buffer_(
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
            for _module in nnet.modules():
                if hasattr(_module, "reset_parameters"):
                    _module.reset_parameters()
        # update nnet
        for _ in tqdm.trange(n_opt_step_per_iter, dynamic_ncols=True, leave=False):
            btmetrics_d: dict[str, float] = _dagger_fit_iter_nnet_selector(
                tstate=tstate, bsz=bsz, plf=plf
            )
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
stdata: thd.TensorDict = compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", mktmpl_cfg.data.name, "nnet_mse_cnnet")
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
        (tdata["xs"].shape[1], None, th.nn.ReLU, None),
        (tdata["xs"].shape[1], None, th.nn.ReLU, None),
    ],
)

# %%
opt = th.optim.Adam(nnet.parameters(), lr=1e-4)

# %%
tstate = _TrainState(
    rplbuf=thrl_data.ReplayBuffer(
        storage=thrl_data.ListStorage(max_size=100_000),
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
    n_iter=1000,
    n_dagger_rollout=128,
    n_opt_step_per_iter=100,
    bsz=1024,
    metrics_func=metrics_func,
    plf=plf,
    vdata=vdata,
    vclassifier=vclassifier,
    eval_every_n_iter=50,
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
