# %%
from __future__ import annotations

import os
from typing import Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels.classifiers
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
PROJ_ROOT: str = mylib.utils.get_project_root_dir()


# %%
# # NOTE mnist
# data_name: str = "mnist"
# _tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=True)
# n_covs: int = _tdata["xs"].shape[1]
# n_labels: int = len(th.unique(_tdata["ys"]))
# max_tdata: Optional[int] = 30000
# _tdata_shuffle_idxs = th.randperm(
#     len(_tdata), generator=th.Generator().manual_seed(279)
# )
# tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
# extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
run_p: str = "experiments/make_template/outputs/big5_cnnet/20251213_024429/0"
run_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, run_p, ".hydra", "config.yaml"))
tafa_cfg = OmegaConf.load(
    os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
)  # type:ignore
data_name: str = tafa_cfg.data.name
_tdata, vdata, tstdata = hd.utils.call(tafa_cfg.data)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
# split training data into two for classifier and afa
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata: thd.TensorDict
extdata: thd.TensorDict
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(mylib.utils.get_project_root_dir(), run_p, "tdata_shuffle_idxs.pt")
)
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
# make classifier
tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
    tafa_cfg.tclassifier,
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
)
if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
    tclassifier_p = os.path.join(
        mylib.utils.get_project_root_dir(), run_p, "tclassifier.pt"
    )
    if os.path.exists(tclassifier_p):
        tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
    else:
        tclassifier.fit_(tmpls)
vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
if tafa_cfg.vclassifier is not None:
    vclassifier = hd.utils.instantiate(
        tafa_cfg.vclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )
    if isinstance(vclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        vclassifier_p = os.path.join(
            mylib.utils.get_project_root_dir(), run_p, "tclassifier.pt"
        )
        if os.path.exists(vclassifier_p):
            vclassifier.load_state_dict(th.load(vclassifier_p, map_location="cpu"))
        else:
            vclassifier.fit_(tmpls)

# %%
init_fidx: int = tafa_cfg.init_fidx
n_tmpls_targ: int = 512
n_cannds_init: int = 5_000
n_cands_targ: int = 5_000
n_cands_mutate: int = 2_500
lmbda: float = 0.055
n_neighs = 100
min_features_targ: int = 1
max_features_targ: Optional[int] = None
n_rounds: int = 20
use_feature_importance_sampling: bool = True
to_update_classifier: bool = False
bsz: int = 128000

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "greedy")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")
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
tmpls = tafalib.makers.templates.make_templates_vanilla(
    tdata=tdata,
    max_tdata=None,
    classifier=tclassifier,
    to_update_classifier=to_update_classifier,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls_targ,
    n_cands=n_cands_targ * n_rounds,
    min_features=min_features_targ,
    max_features=max_features_targ,
    lmbda=lmbda,
    bsz=bsz,
    plf=plf,
)

# %%
tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
    tmpls=tmpls,
    data=tdata,
    classifier=tclassifier,
    lmbda=lmbda,
    bsz=bsz,
    plf=plf,
)
vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: tafalib.functional.knn_cost_est(
        x,
        lmbda=lmbda,
        txs=tdata["xs"],
        tcels=tpcomp["cels"],
        tmpls=tmpls,  # type:ignore
        n_neighs=n_neighs,
        p=2,
    ),
    init_fidx=init_fidx,
    tmpls=tmpls,
    lmbda=lmbda,
    metrics_func=metrics_func,
    plf=plf,
)
plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "eval-val"))

# %%
tfb_logger.finalize("success")
csv_logger.finalize("success")

# %%
