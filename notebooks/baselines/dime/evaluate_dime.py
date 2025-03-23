# %%
from __future__ import annotations

import os
from typing import Any

import dimelib
import hydra as hd
import mylib
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.plugins.environments as pl_plugins_envs
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
from dimelib.utils import MaskLayer
from omegaconf import OmegaConf

# %%
OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
run_p: str = "experiments/baselines/dime/outputs/2025-03-08/big5"

# %%
run_cfg = OmegaConf.load(
    os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
)
output_dir: str = f"outputs/evaluate{run_cfg.data.name}"

# %%
# make dataset
tdata, vdata, tstdata = hd.utils.call(run_cfg.data)
n_covs: int = tdata["xs"].shape[1]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
# make dataloader
n_workers: int | None = os.cpu_count()
if n_workers is None:
    n_workers = 0
tloader = th_data.DataLoader(
    th_data.TensorDataset(tdata["xs"], tdata["ys"]),
    batch_size=run_cfg.bsz,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=n_workers,
)
vloader = th_data.DataLoader(
    th_data.TensorDataset(vdata["xs"], vdata["ys"]),
    batch_size=run_cfg.bsz,
    shuffle=False,
    pin_memory=True,
    num_workers=n_workers,
)

# %%
# make nnet
predictor_nnet: th.nn.Module = hd.utils.instantiate(
    run_cfg.predictor_nnet, in_features=n_covs * 2, out_features=n_labels
)
value_nnet: th.nn.Module = hd.utils.instantiate(
    run_cfg.value_nnet, in_features=n_covs * 2, out_features=n_covs
)
mask_layer = MaskLayer(mask_size=n_covs, append=True)
if run_cfg.share_weights_layer_indices is not None:
    assert isinstance(predictor_nnet, th.nn.Sequential)
    assert isinstance(value_nnet, th.nn.Sequential)
    for _i in run_cfg.share_weights_layer_indices:
        value_nnet[_i] = predictor_nnet[_i]

# %%
# configure loggers
tfb_logger = pl_loggers.TensorBoardLogger(output_dir, name="", version="")
csv_logger = pl_loggers.CSVLogger(save_dir=tfb_logger.log_dir, name="", version="")

# %%
trainer = pl.Trainer(
    logger=[tfb_logger, csv_logger],
    plugins=[pl_plugins_envs.LightningEnvironment()],
)

# %%
cmi_module_kwargs: dict[str, Any] = OmegaConf.to_container(
    run_cfg.cmi_module_cfg.cmi_module
)  # type:ignore
cmi_module_kwargs.pop("_target_")
cmi_module: dimelib.cmi_estimator.CMIEstimator = (
    dimelib.cmi_estimator.CMIEstimator.load_from_checkpoint(
        os.path.join(
            mylib.utils.get_project_root_dir(),
            run_p,
            "checkpoints",
            "best_val_perf_model.ckpt",
        ),
        value_network=value_nnet,
        predictor=predictor_nnet,
        mask_layer=mask_layer,
        **cmi_module_kwargs,
        loss_fn=th.nn.CrossEntropyLoss(reduction="none"),
        val_loss_fn=thm.Accuracy(task="multiclass", num_classes=n_labels),
        map_location="cpu",
    )
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
lmbda = 0.05
inference_out: dict[str, th.Tensor] = cmi_module.inference(
    trainer, vloader, feature_costs=None, lam=lmbda
)

# %%
n_feats: th.Tensor = th.sum(inference_out["mask"], dim=1)
ys: th.Tensor = inference_out["y"]
pyhats: th.Tensor = inference_out["pred"]
metrics_func.reset()
metrics_func.update(pyhats, ys)
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d["feature observed"] = th.mean(n_feats).item()
print(pd.Series(metrics_d))

# %%
