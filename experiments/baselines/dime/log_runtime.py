from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import lightning.fabric.loggers as plf_loggers
import mylib
import pytorch_lightning as pl
import pytorch_lightning.plugins.environments as pl_plugins_envs
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
from dimelib.cmi_estimator import CMIEstimator
from dimelib.utils import MaskLayer
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    train_exp: Optional[MakeTemplateExpConf]
    train_run: Optional[str]
    lmbda: float
    trainer: Any


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


def _get_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "train_run") and cfg.train_run is not None:
        return cfg.train_run
    assert hasattr(cfg, "train_exp") and cfg.train_exp is not None
    return os.path.join(cfg.train_exp.exp_p, str(cfg.train_exp.run_id))


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    run_p: str = _get_run_dir(cfg)
    run_cfg = OmegaConf.load(
        os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
    )
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(run_cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # make dataloader
    # n_workers: int | None = os.cpu_count()
    # if n_workers is None:
    #     n_workers = 0
    n_workers: int | None = 0
    tloader = th_data.DataLoader(
        th_data.TensorDataset(tdata["xs"], tdata["ys"]),
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=n_workers,
    )
    vloader = th_data.DataLoader(
        th_data.TensorDataset(vdata["xs"], vdata["ys"]),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=n_workers,
    )
    tstloader = th_data.DataLoader(
        th_data.TensorDataset(tstdata["xs"], tstdata["ys"]),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=n_workers,
    )
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
    cmi_module_kwargs: dict[str, Any] = OmegaConf.to_container(
        run_cfg.cmi_module_cfg.cmi_module
    )  # type:ignore
    cmi_module_kwargs.pop("_target_")
    cmi_module: CMIEstimator = CMIEstimator.load_from_checkpoint(
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
    # predict with validation
    trainer = pl.Trainer = hd.utils.instantiate(cfg.trainer, _partial_=True)(
        logger=False,
        plugins=[pl_plugins_envs.LightningEnvironment()],
    )
    start_time_ns: int = time.time_ns()
    inference_out: dict[str, th.Tensor] = cmi_module.inference(
        trainer, vloader, feature_costs=None, lam=cfg.lmbda
    )
    end_time_ns: int = time.time_ns()
    # log metrics
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    metrics_func.reset()
    metrics_func.update(
        inference_out["pred"].to(device="cpu"), inference_out["y"].to(device="cpu")
    )
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_d["feature observed"] = th.mean(th.sum(inference_out["mask"], dim=1)).item()
    metrics_d["inference_time_ns"] = end_time_ns - start_time_ns
    metrics_d["avg_pred_time_ns"] = (end_time_ns - start_time_ns) / len(vdata)
    metrics_func.reset()
    tfb_logger.log_metrics(mylib.utils.add_prefix_to_dict(metrics_d, "eval"), step=0)
    csv_logger.log_metrics(mylib.utils.add_prefix_to_dict(metrics_d, "eval"), step=0)
    # finalize logger
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
