# %%
from __future__ import annotations

import os
from typing import Callable, Optional, TypedDict

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

# %%
PROJ_ROOT: str = mylib.utils.get_project_root_dir()

# %%
# mktmpl_run_dir: str = "experiments/make_template/outputs/grid/20250305_145621/2"
# mktmpl_run_dir: str = "experiments/make_template/outputs/cube/20250305_143844/4"
mktmpl_run_dir: str = "experiments/make_template/outputs/big5/20250305_145410/2"

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
cfg = OmegaConf.load(os.path.join(PROJ_ROOT, mktmpl_run_dir, ".hydra", "config.yaml"))

# %%
# make dataset
_tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
_tdata, vdata, tstdata = hd.utils.call(cfg.data)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
# split training data into two for classifier and afa
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
# make classifier
tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
    cfg.tclassifier, xs_train=extdata["xs"].numpy(), ys_train=extdata["ys"].numpy()
)
if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
    tclassifier.fit_(tmpls)
vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
if cfg.vclassifier is not None:
    vclassifier = hd.utils.instantiate(
        cfg.vclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )


# %%
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
def _make_warmup_fit_bsinps(
    bstdata: thd.TensorDict, init_fidx: int, tmpls: th.Tensor
) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bfms: th.Tensor = tmpls[btmplidxs].to(device=bxs.device)
    n_covs: int = bxs.shape[1]
    # randomly drop features
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs), device=bxs.device)
    bnms = th.clamp(bnms - bfms, 0.0, 1.0)
    bnms[:, init_fidx] = 1
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs * bnms, bnms), dim=1)
    return bsinps


class _TrainState(TypedDict):
    nnet: th.nn.Module
    opt: th.optim.Optimizer
    n_trial_itr: int
    n_fit_itr: int
    opt_step: int


def _warmup_fit_iter_nnet_regressor(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> dict[str, float]:
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = _make_warmup_fit_bsinps(
        bstdata=bstdata, init_fidx=init_fidx, tmpls=tmpls
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


def warmup_fit_nnet_regressor(
    tstate: _TrainState,
    stdata: thd.TensorDict,
    init_fidx: int,
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
        bmetrics_d: dict[str, float] = _warmup_fit_iter_nnet_regressor(
            tstate=tstate, bstdata=bstdata, init_fidx=init_fidx, tmpls=tmpls, plf=plf
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
# def _tmplafa_predict(
#     data: thd.TensorDict,
#     classifier: mymodels.classifiers.SubsetFeatureClassifier,
#     cost_est: Callable[[th.Tensor], th.Tensor],
#     init_fidx: int,
#     tmpls: th.Tensor,
#     plf: pl.Fabric,
# ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
#     n_labels: int = classifier.n_labels
#     pyhats: th.Tensor = th.empty((len(data), n_labels), dtype=th.float32)
#     oms: th.Tensor = th.zeros_like(data["xs"])
#     fms: th.Tensor = th.zeros_like(data["xs"])
#     for _i, _data in enumerate(data):
#         _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode(
#             x=_data["xs"],
#             classifier=classifier,
#             cost_est=cost_est,
#             init_fidx=init_fidx,
#             tmpls=tmpls,
#             plf=plf,
#         )
#         pyhats[_i] = _pyhat
#         oms[_i, _fcomb] = 1
#         fms[_i, _fobsd_l] = 1
#     return pyhats, oms, fms


# @th.no_grad()
# def _make_dagger_fit_bsinps(
#     bstdata: thd.TensorDict, boms: th.Tensor, init_fidx: int
# ) -> th.Tensor:
#     # (bsz, n_covs)
#     bxs: th.Tensor = bstdata["xs"]
#     # randomly drop features
#     bnms: th.Tensor = boms * th.randint_like(boms, 0, 2)
#     bnms[:, init_fidx] = 1
#     # (bsz, 2 * n_covs)
#     bsinps: th.Tensor = th.cat((bxs * bnms, bnms), dim=1)
#     return bsinps


def _tmplafa_predict(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, list[list[int]]]:
    n_labels: int = classifier.n_labels
    pyhats: th.Tensor = th.empty((len(data), n_labels), dtype=th.float32)
    oms: th.Tensor = th.zeros_like(data["xs"])
    fms: th.Tensor = th.zeros_like(data["xs"])
    fobsds_l: list[list[int]] = list()
    for _i, _data in enumerate(data):
        _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        pyhats[_i] = _pyhat
        oms[_i, _fcomb] = 1
        fms[_i, _fobsd_l] = 1
        fobsds_l.append(_fobsd_l)
    return pyhats, oms, fms, fobsds_l


@th.no_grad()
def _make_dagger_fit_bsinps(
    bstdata: thd.TensorDict, bfobsds_l: list[list[int]], init_fidx: int
) -> list[th.Tensor]:
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    # make new masks
    bnms_l: list[th.Tensor] = list()
    for _bidx, _fobsd_l in enumerate(bfobsds_l):
        assert _fobsd_l[0] == init_fidx
        _nms_l: list[th.Tensor] = list()
        for _i in range(len(_fobsd_l)):
            _nm: th.Tensor = th.zeros_like(bxs[_bidx])
            _nm[_fobsd_l[: _i + 1]] = 1
            _nms_l.append(_nm)
        bnms_l.append(th.stack(_nms_l))
    # make new selector inputs
    bsinps_l: list[th.Tensor] = list()
    for _bidx, _nm in enumerate(bnms_l):
        _xs: th.Tensor = bxs[_bidx][None, :].expand(len(_nm), -1)
        _sinps: th.Tensor = th.cat((_xs * _nm, _nm), dim=1)
        bsinps_l.append(_sinps)
    return bsinps_l


def _dagger_fit_iter_nnet_regressor(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> dict[str, float]:
    nnet: th.nn.Module = tstate["nnet"].to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    nnet.eval()
    _, _, _, bfobsds_l = _tmplafa_predict(
        data=bstdata,
        classifier=classifier,
        cost_est=lambda x: tafalib.functional.multi_output_nnet_cost_est(
            x, nnet=nnet, lmbda=lmbda, tmpls=tmpls, device=plf.device
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    nnet.train()
    # shape of list (bsz, )
    # shape of tensor in the list (len(boms[_bi]), 2 * n_covs)
    bsinps_l: list[th.Tensor] = _make_dagger_fit_bsinps(
        bstdata=bstdata, bfobsds_l=bfobsds_l, init_fidx=init_fidx
    )
    # # one experience for each datum from the batch
    # # (bsz, 2 * n_covs)
    # bsinps: th.Tensor = th.stack(
    #     [sinps[th.randint(0, len(sinps), ())] for sinps in bsinps_l], dim=0
    # )
    # # (bsz, n_tmpls)
    # bsouts: th.Tensor = nnet(bsinps)
    # # compute selector loss
    # # (bsz, )
    # bslosses: th.Tensor = th.nn.functional.mse_loss(
    #     bsouts, bstdata["cels"], reduction="none"
    # )
    # concatenate all experiences from all data
    # (sum(map(len, bsinps_l)), 2 * n_covs)
    bsinps: th.Tensor = th.cat(bsinps_l, dim=0)
    # (sum(map(len, bsinps_l)), n_tmpls)
    bsouts: th.Tensor = nnet(bsinps)
    # compute selector loss
    # (sum(map(len, bsinps_l)), )
    bstargs: th.Tensor = th.repeat_interleave(
        bstdata["cels"],
        th.tensor([len(sinps) for sinps in bsinps_l], device=plf.device),
        dim=0,
    )
    bslosses: th.Tensor = th.nn.functional.mse_loss(bsouts, bstargs, reduction="none")
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


def dagger_fit_nnet_regressor(
    tstate: _TrainState,
    stdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    n_iter: int,
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
        bstdata: thd.TensorDict = stdata[th.randint(0, len(stdata), (bsz,))].to(
            device=plf.device
        )
        # track metrics
        btmetrics_d: dict[str, float] = _dagger_fit_iter_nnet_regressor(
            tstate=tstate,
            bstdata=bstdata,
            classifier=classifier,
            init_fidx=init_fidx,
            lmbda=lmbda,
            tmpls=tmpls,
            plf=plf,
        )
        pbar.set_postfix(btmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(btmetrics_d, "train_dagger"),
            step=tstate["opt_step"],
        )
        if ckpt_p is not None and (
            (_itr % save_ckpt_every_n_iter) == 0 or (_itr + 1) == n_iter
        ):
            plf.save(os.path.join(ckpt_p, f"dagger_itr{_itr}.pt"), tstate)
        if vdata is not None and (
            _itr % eval_every_n_iter == 0 or (_itr + 1) == n_iter
        ):
            vclassifier = classifier if vclassifier is None else vclassifier
            vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib.functional.multi_output_nnet_cost_est(
                    x,
                    nnet=tstate["nnet"],
                    lmbda=lmbda,
                    tmpls=tmpls,
                    device=plf.device,
                ),
                init_fidx=init_fidx,
                tmpls=tmpls,
                metrics_func=metrics_func,
                plf=plf,
            )
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(vmetrics_d, "eval_dagger"),
                step=tstate["opt_step"],
            )
    pbar.close()
    return tstate


# %%
stdata: thd.TensorDict = compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", cfg.data.name, "nnet_mse")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf_nnet = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

# %%
nnet = mymodels.nn.make_fcn(
    in_features=2 * tdata["xs"].shape[1],
    out_features=len(tmpls),
    layer_specs=[
        (tdata["xs"].shape[1], None, None, None),
        (tdata["xs"].shape[1], None, None, None),
    ],
)

# %%
opt = th.optim.Adam(nnet.parameters(), lr=1e-4)

# %%
tstate = _TrainState(nnet=nnet, opt=opt, n_trial_itr=0, n_fit_itr=0, opt_step=0)

# %%
warmup_fit_nnet_regressor(
    tstate=tstate,
    stdata=stdata,
    init_fidx=cfg.init_fidx,
    tmpls=tmpls,
    n_iter=10_000,
    bsz=1024,
    plf=plf_nnet,
)

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
plf_tmpl = pl.Fabric(accelerator="cpu")

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.knn_cost_est(
        x,
        lmbda=cfg.lmbda,
        txs=tdata["xs"],
        tcels=tpcomp["cels"],
        tmpls=tmpls,
        n_neighs=cfg.n_neighs,
        p=2,
    ),
    init_fidx=cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf_tmpl,
)
print(pd.Series(metrics_d))

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.multi_output_nnet_cost_est(
        x, nnet=nnet, lmbda=cfg.lmbda, tmpls=tmpls, device=plf_nnet.device
    ),
    init_fidx=cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf_tmpl,
)
print(pd.Series(metrics_d))

# %%
dagger_fit_nnet_regressor(
    tstate=tstate,
    stdata=stdata,
    classifier=tclassifier,
    init_fidx=cfg.init_fidx,
    lmbda=cfg.lmbda,
    tmpls=tmpls,
    n_iter=100,
    bsz=1024,
    metrics_func=metrics_func,
    plf=plf_nnet,
    vdata=vdata,
    vclassifier=vclassifier,
    eval_every_n_iter=10,
    ckpt_p=None,
)

# %%
metrics_func.reset()
metrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.multi_output_nnet_cost_est(
        x, nnet=nnet, lmbda=cfg.lmbda, tmpls=tmpls, device=plf_nnet.device
    ),
    init_fidx=cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf_tmpl,
)
print(pd.Series(metrics_d))

# %%
