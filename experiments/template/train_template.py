from dataclasses import dataclass
from typing import Any

import datasets.common
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import torch as th


@dataclass
class TrainTemplateConf:
    data: "_DataConf"
    opt: Any
    model: Any


@dataclass
class _DataConf:
    train_data: datasets.common.DatasetConf
    val_data: datasets.common.DatasetConf = MISSING
    test_data: datasets.common.DatasetConf = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=TrainTemplateConf)


@hydra.main(config_path="./conf", version_base=None)
def main(cfg: TrainTemplateConf):
    data = hydra.utils.call(cfg.data, _recursive_=True)
    model: th.nn.Module = hydra.utils.call(cfg.model)
    opt: th.optim.Optimizer = hydra.utils.instantiate(
        cfg.opt, params=model.parameters()
    )

if __name__ == "__main__":
    main()
