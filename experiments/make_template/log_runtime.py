from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import mymodels
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


@dataclass
class MainConf:
    train_exp: Optional[MakeTemplateExpConf]
    train_run: Optional[str]
    n_instances: int


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


@dataclass
class TAFAMainConf:
    data: Any
    tclassifier: Any
    vclassifier: Optional[Any]
    make_templates_fn: Any
    init_fidx: int
    lmbda: float
    n_neighs: int
    bsz: int
    plf: Any


def _get_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "train_run") and cfg.train_run is not None:
        return cfg.train_run
    assert hasattr(cfg, "train_exp") and cfg.train_exp is not None
    return os.path.join(cfg.train_exp.exp_p, str(cfg.train_exp.run_id))


def main(cfg: MainConf):
    output_dir: str = HydraConfig.get().runtime.output_dir
    tafa_run_p: str = _get_run_dir(cfg)
    tafa_cfg: TAFAMainConf = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), tafa_run_p, ".hydra", "config.yaml"
        )
    )  # type:ignore
    _tdata_shuffle_idxs: th.Tensor = th.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), tafa_run_p, "tdata_shuffle_idxs.pt"
        )
    )
    tmpls: th.Tensor = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), tafa_run_p, "tmpls.pt"),
        weights_only=False,
    )
    tpcomp: thd.TensorDict = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), tafa_run_p, "tpcomp.pt"),
        weights_only=False,
    )
    # make dataset
    _tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    _tdata, vdata, tstdata = hd.utils.call(tafa_cfg.data)
    n_covs: int = _tdata["xs"].shape[1]
    n_labels: int = len(th.unique(_tdata["ys"]))
    # split training data into two for classifier and afa
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata: thd.TensorDict
    extdata: thd.TensorDict
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
            mylib.utils.get_project_root_dir(), tafa_run_p, "tclassifier.pt"
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
                mylib.utils.get_project_root_dir(), tafa_run_p, "tclassifier.pt"
            )
            if os.path.exists(vclassifier_p):
                vclassifier.load_state_dict(th.load(vclassifier_p, map_location="cpu"))
            else:
                vclassifier.fit_(tmpls)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
    plf = pl.Fabric(
        loggers=[tfb_logger, csv_logger],
        accelerator="cpu",
        plugins=plf_plugins_envs.LightningEnvironment(),
    )
    start_time_ns: int = time.time_ns()
    metrics_d: dict[str, float] = tafalib.utils.evaluate(
        data=vdata[
            th.multinomial(th.ones(len(vdata)), cfg.n_instances, replacement=False)
        ],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=tafa_cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=tafa_cfg.n_neighs,
            p=2,
        ),
        init_fidx=tafa_cfg.init_fidx,
        tmpls=tmpls,
        lmbda=tafa_cfg.lmbda,
        metrics_func=thm.MetricCollection(
            thm.Accuracy(task="multiclass", num_classes=n_labels)
        ),
        plf=plf,
    )
    end_time_ns: int = time.time_ns()
    metrics_d["inference_time_ns"] = end_time_ns - start_time_ns
    metrics_d["avg_pred_time_ns"] = (end_time_ns - start_time_ns) / cfg.n_instances
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"), step=0)
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
