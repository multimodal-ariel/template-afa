from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

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


@dataclass
class MainConf:
    data: Any
    tclassifier: Any
    vclassifier: Optional[Any]
    make_templates_fn: Any
    init_fidx: int | Any | None
    lmbda: float
    n_neighs: int
    missing_value: float
    bsz: int
    plf: Any


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    _tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    _tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = _tdata["xs"].shape[1]
    n_labels: int = len(th.unique(_tdata["ys"]))
    # split training data into two for classifier and afa
    _tdata_shuffle_idxs = th.randperm(
        len(_tdata), generator=th.Generator().manual_seed(279)
    )
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
    extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
    # configure logger and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
    plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger],
        plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
    )
    # make classifier
    tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
        cfg.tclassifier, xs_train=extdata["xs"].numpy(), ys_train=extdata["ys"].numpy()
    )
    if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatXGBClassifier):
        import sklearn.exceptions as skl_exceptions
        import sklearn.utils as skl_utils

        try:
            [skl_utils.validation.check_is_fitted(_m) for _m in tclassifier._models]
        except skl_exceptions.NotFittedError:
            tclassifier.fit_(None)
    vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
    if cfg.vclassifier is not None:
        vclassifier = hd.utils.instantiate(
            cfg.vclassifier,
            xs_train=extdata["xs"].numpy(),
            ys_train=extdata["ys"].numpy(),
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
    # add eval validation envirionment kwags
    # TODO modify tafalib.makers.templates direclty to provide common interface
    _mktmpl_fn_kwargs: dict[str, Any] = dict()
    if str.split(cfg.make_templates_fn._target_, ".")[-1] in (
        "make_templates_direct_greedy_with_undo",
        "make_templates_direct_minibatch_gradient",
    ):
        _mktmpl_fn_kwargs.update(
            {
                "n_neighs": cfg.n_neighs,
                "vdata": vdata,
                "vclassifier": vclassifier,
                "metrics_func": metrics_func,
            }
        )
    elif str.split(cfg.make_templates_fn._target_, ".")[-1] in (
        "make_templates_cyclic_greedy_with_undo",
    ):
        _mktmpl_fn_kwargs.update(
            {
                "feature_opt_cycler": hd.utils.instantiate(
                    cfg.make_templates_fn.feature_opt_cycler, _partial_=True
                )(tdata=tdata),
                "n_neighs": cfg.n_neighs,
                "vdata": vdata,
                "vclassifier": vclassifier,
                "metrics_func": metrics_func,
            }
        )
        del cfg.make_templates_fn.feature_opt_cycler
    # make templates
    make_templates_fn: Callable = hd.utils.instantiate(
        cfg.make_templates_fn, _partial_=True
    )
    init_fidx: int | Callable | None = cfg.init_fidx
    if cfg.init_fidx is not None and not isinstance(cfg.init_fidx, int):
        init_fidx = hd.utils.instantiate(cfg.init_fidx, _partial_=True)
    tmpls: th.Tensor = make_templates_fn(
        tdata=tdata,
        missing_value=cfg.missing_value,
        classifier=tclassifier,
        init_fidx=init_fidx,
        lmbda=cfg.lmbda,
        bsz=cfg.bsz,
        plf=plf,
        **_mktmpl_fn_kwargs,
    )
    # save templates
    th.save(_tdata_shuffle_idxs, os.path.join(output_dir, "tdata_shuffle_idxs.pt"))
    th.save(tmpls, os.path.join(output_dir, "tmpls.pt"))
    # save training classifier
    if len(tclassifier.state_dict()) != 0:
        th.save(tclassifier.state_dict(), os.path.join(output_dir, "tclassifier.pt"))
    # evaluate validation set performance
    tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls_with_missing_feature(
        tmpls=tmpls,
        data=tdata,
        missing_value=cfg.missing_value,
        classifier=tclassifier,
        lmbda=cfg.lmbda,
        bsz=cfg.bsz,
        plf=plf,
    )
    th.save(tpcomp, os.path.join(output_dir, "tpcomp.pt"))
    metrics_d: dict[str, float] = tafalib.utils.evaluate_with_missing_feature(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=cfg.n_neighs,
            p=2,
        ),
        init_fidx=(
            init_fidx
            if isinstance(init_fidx, int)
            else int(th.argwhere(th.all(tmpls.to(dtype=th.bool), dim=0).flatten())[0])
        ),
        tmpls=tmpls,
        missing_value=cfg.missing_value,
        lmbda=cfg.lmbda,
        metrics_func=metrics_func,
        plf=plf,
    )
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"))
    # save validation time classifier
    # if it's neither the one used in training time
    # nor has empty state_dict
    if vclassifier is not tclassifier and len(vclassifier.state_dict()) != 0:
        th.save(vclassifier.state_dict(), os.path.join(output_dir, "vclassifier.pt"))
    # logger flush record and close
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
