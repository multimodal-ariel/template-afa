# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import numpy as np
import pandas as pd
import sklearn.feature_selection as skl_fs
import tensordict as thd
import torch as th
import torchmetrics as thm
from omegaconf import OmegaConf


# %%
@dataclass
class MainConf:
    data: Any
    n_feats: int
    classifier: Optional[Any]


# %%
cfg: MainConf = OmegaConf.load("test.yaml")  # type:ignore
output_dir: str = f"outputs/{cfg.data.name}/nfeats{cfg.n_feats}"
os.makedirs(output_dir, exist_ok=True)

# %%
tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
tdata, vdata, tstdata = hd.utils.call(cfg.data)
n_covs: int = tdata["xs"].shape[1]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
slctr = skl_fs.SelectKBest(k=cfg.n_feats)
txs_n: np.ndarray = slctr.fit_transform(tdata["xs"].numpy(), tdata["ys"].numpy())

# %%
classifier = hd.utils.instantiate(cfg.classifier)
classifier.fit(txs_n, tdata["ys"].numpy())

# %%
th.save(
    {"classifier": classifier, "slctr": slctr}, os.path.join(output_dir, "models.pt")
)

# %%
vxs_n: np.ndarray = slctr.transform(vdata["xs"].numpy())  # type:ignore
vpyhats_n: np.ndarray = classifier.predict_proba(vxs_n)

# %%
# configure logger and ckpt path
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
plf = pl.Fabric(
    accelerator="cpu",
    loggers=[tfb_logger, csv_logger],
    plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
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

# %%
metrics_func.reset()
metrics_func.update(th.as_tensor(vpyhats_n, dtype=th.float32), vdata["ys"])
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_d.update({"feature observed": cfg.n_feats})
metrics_func.reset()
plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_val"))
tfb_logger.finalize("success")
csv_logger.finalize("success")

# %%
print(pd.Series(metrics_d))

# %%
