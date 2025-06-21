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
import xgboost as xgbst
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
@th.no_grad()
def compile_cross_entropy_regressor_dataset(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    tmpls: th.Tensor,
    n_pairs_per_sample: int,
    plf: pl.Fabric,
    rg: Optional[th.Generator] = None,
) -> thd.TensorDict:
    classifier.to(device=plf.device)
    n_covs: int = tdata["xs"].shape[1]
    n_tmpls: int = len(tmpls)
    sinps: th.Tensor = th.empty(
        (len(tdata), n_pairs_per_sample, 3 * tdata["xs"].shape[1]), dtype=th.float32
    )
    stargs: th.Tensor = th.empty((len(tdata), n_pairs_per_sample), dtype=th.float32)
    for _i in tqdm.trange(
        len(tdata), desc="make_fit_dataset", leave=False, dynamic_ncols=True
    ):
        _x: th.Tensor = tdata[_i]["xs"]
        _y: th.Tensor = tdata[_i]["ys"]
        _fmidxs: th.Tensor = th.randint(
            0, n_tmpls, (2, n_pairs_per_sample), generator=rg
        )
        # (n_pairs_per_sample, n_covs)
        # ms1 are "simulated" current observations
        _boms: th.Tensor = (
            tmpls[_fmidxs[0]]
            - th.randint(0, 2, (n_pairs_per_sample, n_covs), generator=rg)
        ).clamp_(0, 1)
        _boms[:, init_fidx] = 1
        # ms2 are from feature templates
        _btmpls: th.Tensor = tmpls[_fmidxs[1]]
        # observed features at the end of the episode
        _bfms: th.Tensor = th.maximum(_boms, _btmpls)
        # use end of episode observed feature to make inference
        _bxs: th.Tensor = _x[None, :].expand(n_pairs_per_sample, -1)
        # (n_pairs_per_sample, )
        _bys: th.Tensor = _y.expand(n_pairs_per_sample)
        # (n_pairs_per_sample, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(
            ctxs=_bxs.to(device=plf.device), acts=_bfms.to(device=plf.device)
        ).to(device="cpu")
        # compute cross-entropy
        # (n_pairs_per_sample, )
        _bcels: th.Tensor = th.nn.functional.nll_loss(
            th.log(_bpyhats), _bys, reduction="none"
        )
        # update sinps stargs
        sinps[_i] = th.cat((_bxs, _boms, _btmpls), dim=1)
        stargs[_i] = _bcels
    stdata = thd.TensorDict(
        {
            "sinps": sinps.flatten(0, 1),
            "stargs": stargs.flatten(0, 1),
        }
    ).auto_batch_size_(1)
    return stdata


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
# stdata: thd.TensorDict = compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)
# sxs_n, sys_n = make_xgb_ce_dataset(
#     xs=tdata["xs"],
#     ys=tdata["ys"],
#     tmpls=tmpls,
#     classifier=tclassifier,
#     init_fidx=mktmpl_cfg.init_fidx,
#     n_pairs_per_sample=256,
# )

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
stdata = compile_cross_entropy_regressor_dataset(
    tdata,
    classifier=tclassifier,
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    n_pairs_per_sample=256,
    plf=plf,
    rg=None,
)

# %%
xgbr_kwargs = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "tree_method": "hist",
    "random_state": 42,
}
xgbr = xgbst.XGBRegressor(**xgbr_kwargs)
xgbr.fit(stdata["sinps"].numpy(), stdata["stargs"].numpy())

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
    cost_est=lambda ctx: tafalib.functional.multi_output_sklearn_cost_est(
        ctx, sklr=xgbr, lmbda=mktmpl_cfg.lmbda, tmpls=tmpls
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
