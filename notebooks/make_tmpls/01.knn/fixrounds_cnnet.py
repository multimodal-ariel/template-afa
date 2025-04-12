# %%
from __future__ import annotations

import os
from typing import Optional

import _classifiers
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mymodels.classifiers
import pandas as pd
import tafalib.functional
import tafalib.makers.templates
import tafalib.utils
import tensordict as thd
import torch as th
import torchmetrics as thm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = "../../../"

# %%
# # NOTE cube
# data_name: str = "cube_20_0.3"
# max_tdata: Optional[int] = None
# tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# n_covs: int = tdata["xs"].shape[1]
# n_labels: int = len(th.unique(tdata["ys"]))
# tclassifier = _classifiers.SubsetFeatureNaiveBayes(
#     0.3, xs_train=tdata["xs"].numpy(), ys_train=tdata["ys"].numpy()
# )
# vclassifier = tclassifier
# init_fidx: int = 6
# n_tmpls_targ: int = 64
# n_cands_targ: int = 10_000
# min_features_targ: int = 1
# max_features_targ: Optional[int] = None
# min_features_init: int = 10
# feature_decrement: int = 2
# use_feature_importance_sampling: bool = True
# lmbda: float = 0.3
# bsz: int = 100_000_000

# %%
# NOTE big5
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
max_tdata: Optional[int] = None
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
run_p: str = (
    "experiments/pretrain/nnet_subset_feature_classifier/outputs/big5/20250312_224514"
)
run_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, run_p, ".hydra", "config.yaml"))
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
        state_dict_p=os.path.join(PROJ_ROOT, run_p, "classifier.pt"),
    )
)
tclassifier.fit_kwargs["n_iter"] = 1000
tclassifier.fit_kwargs["bsz"] = 8192
vclassifier = tclassifier
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

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "fixrounds_cnnet")
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
    )


# %%
tmpls = tafalib.makers.templates.make_templates_fix_rounds(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    to_update_classifier=True,
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
)

# %%
tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
    tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz, plf=plf
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
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=10,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
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
        "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
        "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
    }
)
print(pd.Series(metrics_d))


# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=10,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
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
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=tclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=10,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
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
