from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import numpy as np
import sklearn.feature_selection as skl_fs
import tensordict as thd
import torch as th
import torchmetrics as thm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    data: Any
    n_feats: int
    classifier: Optional[Any]


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


def main(cfg: MainConf):
    output_dir: str = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # load data
    tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # configure lightning
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
    # select features
    slctr = skl_fs.SelectKBest(k=cfg.n_feats)
    txs_n: np.ndarray = slctr.fit_transform(tdata["xs"].numpy(), tdata["ys"].numpy())
    # train classifier
    classifier = hd.utils.instantiate(cfg.classifier)
    classifier.fit(txs_n, tdata["ys"].numpy())
    th.save(
        {"classifier": classifier, "slctr": slctr},
        os.path.join(output_dir, "models.pt"),
    )
    # validate using subset feature classifier
    vxs_n: np.ndarray = slctr.transform(vdata["xs"].numpy())  # type:ignore
    vpyhats_n: np.ndarray = classifier.predict_proba(vxs_n)
    # compute metrics
    metrics_func.reset()
    metrics_func.update(th.as_tensor(vpyhats_n, dtype=th.float32), vdata["ys"])
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_d.update({"feature observed": cfg.n_feats})
    metrics_func.reset()
    # log metrics
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_val"))
    tfb_logger.finalize("success")
    csv_logger.finalize("success")


if __name__ == "__main__":

    @hd.main(version_base=None)
    def _main(cfg: MainConf):
        logger = logging.getLogger(HydraConfig.get().job.name)
        try:
            main(cfg)
        except Exception as e:
            logger.error(e, exc_info=True, stack_info=True)
            traceback.print_exception(e)

    _main()
