from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any

import diff_models_original
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import mymodels
import omegaconf as omgcf
import pandas as pd
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
from hydra.core.hydra_config import HydraConfig

th.set_grad_enabled(False)


@dataclass
class MainConf:
    data: Any
    lmbda_csv_p: str
    ckpt_cfg: CkptConf
    n_neighs: int
    bsz: int
    n_instances: int
    plf: Any


@dataclass
class CkptConf:
    dir: str
    file_stub: str


def load_from_ckpt(
    tdata: thd.TensorDict, data_name: str, ckpt_p: str
) -> tuple[int, mymodels.classifiers.SubsetFeatureClassifier, th.Tensor]:
    data_name_d = {
        "big5_C_cls": "big5",
        "cube_20_0.3": "cube",
        "engine-fualt": "engine",
        "fashion-mnist": "fashionfull",
        "fashion-mnist-16x16": "fashion",
        "grid_data": "grid",
        "gas": "gas",
        "mnist": "mnist",
    }
    hidden_size_d = {
        "mnist": 512,
        "big5": 256,
        "grid": 256,
        "gas": 256,
        "cube": 256,
        "fashion": 512,
        "engine": 256,
    }
    state_dict = th.load(ckpt_p, weights_only=False, map_location="cpu")
    _tmp = dict(
        map(
            lambda _kv: (_kv[0][10:], _kv[1]),
            filter(
                lambda _kv: str.startswith(_kv[0], "surrogate."), state_dict.items()
            ),
        )
    )
    _nnet_tmp = diff_models_original.SurrogateClassifier(
        input_dim=tdata["xs"].shape[1],
        n_classes=len(th.unique(tdata["ys"])),
        hidden_dim=hidden_size_d[data_name_d[data_name]],
        dataset_name=data_name_d[data_name],
    )
    _nnet_tmp.load_state_dict(_tmp)
    classifier = mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier(
        _nnet_tmp.net,
        xs_train=tdata["xs"].numpy(force=True),
        ys_train=tdata["ys"].numpy(force=True),
        fit_kwargs=None,
    )
    tmpls: th.Tensor = th.clamp_(
        th.where(th.sigmoid(state_dict["template_logits"]) < 0.5, 0, 1)
        + state_dict["start_mask"],
        0.0,
        1.0,
    ).to(dtype=th.long)
    init_fidx: int = int(th.argwhere(state_dict["start_mask"].flatten() == 1).item())
    return init_fidx, classifier, tmpls


def run(
    tdata: thd.TensorDict,
    vdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    tmpls: th.Tensor,
    lmbda: float,
    n_neighs: int,
    bsz: int,
    metrics_fn: thm.MetricCollection,
    plf: pl.Fabric,
):
    classifier.eval()
    # evaluate validation set performance
    tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
        tmpls=tmpls,
        data=tdata,
        classifier=classifier,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )
    metrics_d: dict[str, float] = tafalib.utils.evaluate(
        data=vdata,
        classifier=classifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=n_neighs,
            p=2,
        ),
        init_fidx=(
            init_fidx
            if isinstance(init_fidx, int)
            else int(th.argwhere(th.all(tmpls.to(dtype=th.bool), dim=0).flatten())[0])
        ),
        tmpls=tmpls,
        lmbda=lmbda,
        metrics_func=metrics_fn,
        plf=plf,
    )
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"))


def main(cfg: MainConf):
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_labels: int = len(th.unique(tdata["ys"]))
    # read lambdas used to train gumbel policy
    lmbdas_d: dict[int, float] = (
        pd.read_csv(os.path.join(mylib.utils.get_project_root_dir(), cfg.lmbda_csv_p))
        .set_index("idx")
        .to_dict()["alpha"]
    )
    #
    # loop thorugh all gumbel policies
    metrics_fn = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    for _run_id, _lmbda in tqdm.tqdm(
        lmbdas_d.items(),
        desc="run",
        dynamic_ncols=True,
        leave=True,
    ):
        # prep lightnign fabric and loggers
        _tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version=None)  # type: ignore
        _csv_logger = plf_loggers.CSVLogger(root_dir=_tfb_logger.log_dir, name="", version="")  # type: ignore
        _plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
            loggers=[_tfb_logger, _csv_logger],
            plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
        )
        _run_p: str = _tfb_logger.log_dir
        os.makedirs(_run_p, exist_ok=True)
        # load ckpt
        _init_fidx, _classifier, _tmpls = load_from_ckpt(
            tdata=tdata,
            data_name=cfg.data.name,
            ckpt_p=os.path.join(
                mylib.utils.get_project_root_dir(),
                cfg.ckpt_cfg.dir,
                str.format(cfg.ckpt_cfg.file_stub, _run_id),
            ),
        )
        th.save(_tmpls, os.path.join(_run_p, "tmpls.pt"))
        # evaluate validation set performance
        tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
            tmpls=_tmpls,
            data=tdata,
            classifier=_classifier,
            lmbda=_lmbda,
            bsz=cfg.bsz,
            plf=_plf,
        )
        th.save(tpcomp, os.path.join(_run_p, "tpcomp.pt"))
        metrics_d: dict[str, float] = tafalib.utils.evaluate(
            data=(
                vdata
                if (not hasattr(cfg, "n_instances")) or cfg.n_instances is None
                else vdata[
                    th.multinomial(
                        th.ones(len(vdata)),
                        min(cfg.n_instances, len(vdata)),
                        replacement=False,
                    )
                ]
            ),
            classifier=_classifier,
            cost_est=lambda x: tafalib.functional.knn_cost_est(
                x,
                lmbda=_lmbda,
                txs=tdata["xs"],
                tcels=tpcomp["cels"],
                tmpls=_tmpls,
                n_neighs=cfg.n_neighs,
                p=2,
            ),
            init_fidx=(
                _init_fidx
                if isinstance(_init_fidx, int)
                else int(
                    th.argwhere(th.all(_tmpls.to(dtype=th.bool), dim=0).flatten())[0]
                )
            ),
            tmpls=_tmpls,
            lmbda=_lmbda,
            metrics_func=metrics_fn,
            plf=_plf,
        )
        _plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"))
        _tfb_logger.finalize("success")
        _csv_logger.finalize("success")


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
