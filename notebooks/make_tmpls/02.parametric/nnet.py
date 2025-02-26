# %%
from __future__ import annotations

import os
from typing import Optional, TypedDict

import _classifiers
import _tmplfns
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import pandas as pd
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm


# %%
def nnet_cost_est(
    inps: th.Tensor,
    nnet: th.nn.Module,
    lmbda: float,
    tmpls: th.Tensor,
    device: th.device,
) -> th.Tensor:
    nnet.to(device=device)
    # (n_tmpls, n_covs)
    tmpls = tmpls.to(device=device)
    # (n, n_covs)
    inps = inps.to(device=device)
    fms: th.Tensor = inps[n_covs:]
    # (n, n_tmpls)
    cels: th.Tensor = nnet(inps)
    # (n, n_tmpls, n_covs)
    fms_avail: th.Tensor = th.maximum(
        tmpls[None, :, :] - fms[:, None, :], th.as_tensor(0.0, device=device)
    )
    costs: th.Tensor = cels + lmbda * th.sum(fms_avail, dim=2)
    return costs


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
def _make_fit_bsinps(bstdata: thd.TensorDict, tmpls: th.Tensor) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bfms: th.Tensor = tmpls[btmplidxs].to(device=bxs.device)
    n_covs: int = bxs.shape[1]
    # randomly drop features
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs))
    bnms = th.where(bfms == 0, 0, bnms)
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs * bnms, bnms), dim=1)
    return bsinps


class _TrainState(TypedDict):
    nnet: th.nn.Module
    opt: th.optim.Optimizer
    n_trial_itr: int
    n_fit_itr: int
    opt_step: int


def _fit_iter(
    tstate: _TrainState,
    tloader: th_data.DataLoader,
    tmpls: th.Tensor,
    pbar: tqdm.tqdm,
    plf: pl.Fabric,
) -> dict[str, float]:
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    slosses_l: list[th.Tensor] = list()
    for bstdata in tloader:
        bstdata: thd.TensorDict
        bstdata = bstdata.to(device=plf.device)
        # (bsz, 2 * n_covs)
        bsinps: th.Tensor = _make_fit_bsinps(bstdata=bstdata, tmpls=tmpls)
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
        slosses_l.append(bslosses.detach().to(device="cpu"))
        bmetrics_d: dict[str, float] = {
            "bsloss": bsloss.item(),
        }
        pbar.set_postfix(bmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(bmetrics_d, "train"), step=tstate["opt_step"]
        )
        tstate["opt_step"] = tstate["opt_step"] + 1
    sloss_avg: th.Tensor = th.mean(th.cat(slosses_l, dim=0))
    metrics_d: dict[str, float] = {"sloss_avg": sloss_avg.item()}
    return metrics_d


def fit(
    tstate: _TrainState,
    stdata: thd.TensorDict,
    tmpls: th.Tensor,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
):
    tloader = th_data.DataLoader(
        stdata,  # type: ignore
        batch_size=bsz,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _ in pbar:
        metrics_d: dict[str, float] = _fit_iter(
            tstate=tstate,
            tloader=tloader,
            tmpls=tmpls,
            pbar=pbar,
            plf=plf,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "train"), step=tstate["n_fit_itr"]
        )
        tstate["n_fit_itr"] = tstate["n_fit_itr"] + 1
    pbar.close()


# %%
# NOTE cube
# data_name: str = "cube_20_0.3"
# max_tdata: Optional[int] = None
# tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# n_covs: int = tdata["xs"].shape[1]
# n_labels: int = len(th.unique(tdata["ys"]))
# classifier = SubsetFeatureNaiveBayes(
#     0.3, xs_train=tdata["xs"].numpy(), ys_train=tdata["ys"].numpy()
# )
# metrics_func = thm.MetricCollection(
#     {
#         "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
#         "precision": thm.Precision(task="multiclass", num_classes=n_labels),
#         "recall": thm.Recall(task="multiclass", num_classes=n_labels),
#         "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
#         "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
#     }
# )
# init_fidx: int = 6
# n_tmpls_targ: int = 64
# n_cands_targ: int = 10_000
# min_features_targ: int = 1
# max_features_targ: Optional[int] = None
# min_features_init: int = 10
# feature_decrement: int = 2
# lmbda: float = 0.3
# bsz: int = 1024

# %%
# NOTE big5
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
max_tdata: Optional[int] = 8192
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
tclassifier = _classifiers.SubsetFeatureConcatXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgb_kwargs={"n_estimators": 40},
    fraction_training_data_per_split=1.0,
    n_splits=64,
    n_tmpl_per_instance=4,
)
vclassifier = mymodels.classifiers.SubsetFeatureXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgbc_kwargs={"n_estimators": 40},
)
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)
init_fidx: int = 35
n_tmpls_targ: int = 128
n_cands_targ: int = 10_000
lmbda: float = 0.075
n_neighs: int = 10
min_features_targ: int = 1
max_features_targ: Optional[int] = None
n_rounds: int = 3
feature_decrement: int = 2
use_feature_importance_sampling: bool = True
bsz: int = 8192

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "vanilla")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf_tmpl = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="cpu")
plf_nnet = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

# %%
if init_fidx is None:
    init_fidx, bestfm = _tmplfns.identify_init_fidx(
        tdata=tdata,
        classifier=tclassifier,
        max_features=max_features_targ,
        n_repeat=2,
        n_iter=500,
        lmbda=lmbda,
        bsz=bsz,
    )

# %%
tmpls = _tmplfns.make_templates_vanilla(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls_targ,
    n_cands=n_cands_targ,
    min_features=1,
    max_features=max_features_targ,
    lmbda=lmbda,
    bsz=bsz,
    plf=plf_tmpl,
)

# %%
tpcomp: thd.TensorDict = _tmplfns.precomp_rwds_for_tmpls(
    tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz
)

# %%
nnet = mymodels.nn.make_fcn(
    in_features=2 * tdata["xs"].shape[1],
    out_features=len(tmpls),
    layer_specs=[
        (tdata["xs"].shape[1], None, None, None),
        (tdata["xs"].shape[1], None, None, None),
    ],
)
opt = th.optim.Adam(nnet.parameters())

# %%
fit(
    tstate=_TrainState(nnet=nnet, opt=opt, n_trial_itr=0, n_fit_itr=0, opt_step=0),
    stdata=compile_selector_dataset(tdata, tpcomp),
    tmpls=tmpls,
    n_iter=1000,
    bsz=1024,
    plf=plf_nnet,
)

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: _tmplfns.knn_cost_est(
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
        plf=plf_tmpl,
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
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: nnet_cost_est(
            x, nnet=nnet, lmbda=lmbda, tmpls=tmpls, device=plf_nnet.device
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf_tmpl,
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
