from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    mktmpl_exp: Optional[MakeTemplateExpConf]
    mktmpl_run: Optional[str]
    n_pairs_per_sample: int
    sklr: Any
    plf: Any


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


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


def _get_mktmpl_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "mktmpl_run") and cfg.mktmpl_run is not None:
        return cfg.mktmpl_run
    assert hasattr(cfg, "mktmpl_exp") and cfg.mktmpl_exp is not None
    return os.path.join(cfg.mktmpl_exp.exp_p, str(cfg.mktmpl_exp.run_id))


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # get experiment dir
    mktmpl_run_dir: str = _get_mktmpl_run_dir(cfg)
    # load saved result from make_templates
    tmpls: th.Tensor = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tmpls.pt"),
        weights_only=False,
    )
    mktmpl_cfg = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), mktmpl_run_dir, ".hydra", "config.yaml"
        )
    )
    OmegaConf.save(mktmpl_cfg, os.path.join(output_dir, ".hydra", "mktmpl_config.yaml"))
    # make dataset
    _tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    _tdata, vdata, tstdata = hd.utils.call(mktmpl_cfg.data)
    n_covs: int = _tdata["xs"].shape[1]
    n_labels: int = len(th.unique(_tdata["ys"]))
    # split training data into two for classifier and afa
    _tdata_shuffle_idxs: th.Tensor = th.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tdata_shuffle_idxs.pt"
        )
    )
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
    extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
    # make classifier
    tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
        mktmpl_cfg.tclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )
    if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        tclassifier_p = os.path.join(
            mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
        )
        if os.path.exists(tclassifier_p):
            tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
        else:
            tclassifier.fit_(tmpls)
    vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
    if mktmpl_cfg.vclassifier is not None:
        vclassifier = hd.utils.instantiate(
            mktmpl_cfg.vclassifier,
            xs_train=extdata["xs"].numpy(),
            ys_train=extdata["ys"].numpy(),
        )
        if isinstance(vclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            vclassifier_p = os.path.join(
                mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
            )
            if os.path.exists(vclassifier_p):
                vclassifier.load_state_dict(th.load(vclassifier_p, map_location="cpu"))
            else:
                vclassifier.fit_(tmpls)
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
    ckpt_p: str = os.path.join(tfb_logger.log_dir, "checkpoints")
    plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger]
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
    # make nnet
    sklr: mymodels.protocols.ModuleHasPredict = hd.utils.instantiate(cfg.sklr)
    # preapre fitting cost estimator
    stdata: thd.TensorDict = compile_cross_entropy_regressor_dataset(
        tdata=tdata,
        classifier=tclassifier,
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        n_pairs_per_sample=cfg.n_pairs_per_sample,
        plf=plf,
    )
    # warmup nnet cost estimator
    sklr.fit(stdata["sinps"].numpy(), stdata["stargs"].numpy())
    th.save(
        {
            "sklr": sklr,
            "stdata": stdata,
        },
        os.path.join(ckpt_p, "warmup_end.pt"),
    )
    # evaluate validation set performance with nnet after warmup
    metrics_d = tafalib.utils.evaluate(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.multi_output_sklearn_cost_est(
            x, sklr=sklr, lmbda=mktmpl_cfg.lmbda, tmpls=tmpls
        ),
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf,
    )
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"))
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
