from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.plugins.environments as pl_plugins_envs
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
from dimelib.utils import MaskLayer
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    data: Any
    bsz: int
    predictor_nnet: Any
    value_nnet: Any
    share_weights_layer_indices: Optional[list[int]]
    pretrain_module_cfg: MaskingModuleConf
    cmi_module_cfg: CMIModuleConf


@dataclass
class MaskingModuleConf:
    masking_module: Any
    masking_trainer: Any


@dataclass
class CMIModuleConf:
    cmi_module: Any
    cmi_trainer: Any


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # make dataloader
    n_workers: int | None = os.cpu_count()
    if n_workers is None:
        n_workers = 0
    tloader = th_data.DataLoader(
        th_data.TensorDataset(tdata["xs"], tdata["ys"]),
        batch_size=cfg.bsz,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=n_workers,
    )
    vloader = th_data.DataLoader(
        th_data.TensorDataset(vdata["xs"], vdata["ys"]),
        batch_size=cfg.bsz,
        shuffle=False,
        pin_memory=True,
        num_workers=n_workers,
    )
    tstloader = th_data.DataLoader(
        th_data.TensorDataset(tstdata["xs"], tstdata["ys"]),
        batch_size=cfg.bsz,
        shuffle=False,
        pin_memory=True,
        num_workers=n_workers,
    )
    # make nnet
    predictor_nnet: th.nn.Module = hd.utils.instantiate(
        cfg.predictor_nnet, in_features=n_covs * 2, out_features=n_labels
    )
    value_nnet: th.nn.Module = hd.utils.instantiate(
        cfg.value_nnet, in_features=n_covs * 2, out_features=n_covs
    )
    mask_layer = MaskLayer(mask_size=n_covs, append=True)
    if cfg.share_weights_layer_indices is not None:
        assert isinstance(predictor_nnet, th.nn.Sequential)
        assert isinstance(value_nnet, th.nn.Sequential)
        for _i in cfg.share_weights_layer_indices:
            value_nnet[_i] = predictor_nnet[_i]
    # configure loggers
    tfb_logger = pl_loggers.TensorBoardLogger(output_dir, name="", version="")
    csv_logger = pl_loggers.CSVLogger(save_dir=tfb_logger.log_dir, name="", version="")
    # pretrain predictor_nnet
    pretrain_module: pl.LightningModule = hd.utils.instantiate(
        cfg.pretrain_module_cfg.masking_module,
        model=predictor_nnet,
        mask_layer=mask_layer,
        loss_fn=th.nn.CrossEntropyLoss(),
        val_loss_fn=thm.Accuracy(task="multiclass", num_classes=n_labels),
    )
    pretrain_trainer: pl.Trainer = hd.utils.instantiate(
        cfg.pretrain_module_cfg.masking_trainer, _partial_=True
    )(
        logger=[tfb_logger, csv_logger],
        plugins=[pl_plugins_envs.LightningEnvironment()],
    )
    pretrain_trainer.fit(pretrain_module, tloader, vloader)
    # configure ckpt
    ckpt_callback = pl_callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="Perf Val/Final",
        mode="min",
        filename="best_val_perf_model",
        verbose=False,
    )
    # train cmi estimator
    cmi_module: pl.LightningModule = hd.utils.instantiate(
        cfg.cmi_module_cfg.cmi_module,
        value_network=value_nnet,
        predictor=predictor_nnet,
        mask_layer=mask_layer,
        loss_fn=th.nn.CrossEntropyLoss(reduction="none"),
        val_loss_fn=thm.Accuracy(task="multiclass", num_classes=n_labels),
    )
    cmi_trainer: pl.Trainer = hd.utils.instantiate(
        cfg.cmi_module_cfg.cmi_trainer, _partial_=True
    )(
        logger=[tfb_logger, csv_logger],
        callbacks=[ckpt_callback],
        plugins=[pl_plugins_envs.LightningEnvironment()],
    )
    cmi_trainer.fit(cmi_module, tloader, vloader)
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
