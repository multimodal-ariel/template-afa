from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import _daggerlib
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import torchrl.data as thrl_data
import tqdm.auto as tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    mktmpl_exp: Optional[MakeTemplateExpConf]
    mktmpl_run: Optional[str]
    max_rplbuf_size: int
    nnet: Any
    warmup_nnet_tcfg: WarmupNeuralNetTrainConf
    dagger_nnet_tcfg: DaggerNeuralNetTrainConf
    plf: Any


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


@dataclass
class WarmupNeuralNetTrainConf:
    opt: Any
    n_fit_iter: int
    bsz: int
    save_ckpt_every_n_iter: int


@dataclass
class DaggerNeuralNetTrainConf:
    opt: Any
    to_reset_nnet_after_dagger_rollout: bool
    n_fit_iter: int
    n_dagger_rollout: int
    n_opt_step_per_iter: int
    bsz: int
    eval_every_n_iter: int
    save_ckpt_every_n_iter: int


class _TrainState(TypedDict):
    rplbuf: thrl_data.ReplayBuffer
    nnet: th.nn.Module
    opt: th.optim.Optimizer
    fit_itr: int
    opt_step: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


def warmup_fit_nnet_selector(
    tstate: _daggerlib._TrainState,
    stdata: thd.TensorDict,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
    ckpt_p: Optional[str] = None,
    save_ckpt_every_n_iter: int = 1,
):
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _itr in pbar:
        bstdata: thd.TensorDict = stdata[th.randint(0, len(stdata), (bsz,))].to(
            device=plf.device
        )
        # track metrics
        bmetrics_d: dict[str, float] = _daggerlib._warmup_fit_iter_nnet_selector(
            tstate=tstate,
            bstdata=bstdata,
            init_fidx=init_fidx,
            lmbda=lmbda,
            tmpls=tmpls,
            plf=plf,
        )
        pbar.set_postfix(bmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(bmetrics_d, "train_warmup"),
            step=tstate["opt_step"],
        )
        if ckpt_p is not None and (
            (_itr % save_ckpt_every_n_iter) == 0 or (_itr + 1) == n_iter
        ):
            plf.save(os.path.join(ckpt_p, f"warmup_itr{_itr}.pt"), tstate)
    pbar.close()
    return tstate


def dagger_fit_nnet_selector(
    tstate: _daggerlib._TrainState,
    stdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    to_reset_nnet_after_dagger_rollout: bool,
    n_iter: int,
    n_dagger_rollout: int,
    n_opt_step_per_iter: int,
    bsz: int,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
    vdata: Optional[thd.TensorDict] = None,
    vclassifier: Optional[mymodels.classifiers.SubsetFeatureClassifier] = None,
    eval_every_n_iter: int = 1,
    ckpt_p: Optional[str] = None,
    save_ckpt_every_n_iter: int = 1,
):
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _itr in pbar:
        n_rollout: int = min(n_dagger_rollout, len(stdata))
        bstdata: thd.TensorDict = stdata[
            th.multinomial(th.ones((len(stdata),)), n_rollout, replacement=False)
        ]
        # dataset aggregation (dagger)
        _daggerlib._update_dagger_replay_buffer_(
            tstate=tstate,
            bstdata=bstdata,
            classifier=classifier,
            init_fidx=init_fidx,
            lmbda=lmbda,
            tmpls=tmpls,
            plf=plf,
        )
        # if we want to train from scratch, then we need to re-initialize the weight
        if to_reset_nnet_after_dagger_rollout:
            for _module in tstate["nnet"].modules():
                if hasattr(_module, "reset_parameters"):
                    _module.reset_parameters()
        # update nnet
        for _ in tqdm.trange(n_opt_step_per_iter, dynamic_ncols=True, leave=False):
            btmetrics_d: dict[str, float] = _daggerlib._dagger_fit_iter_nnet_selector(
                tstate=tstate, bsz=bsz, plf=plf
            )
            # track metrics
            pbar.set_postfix(btmetrics_d)
            btmetrics_d["fit_itr"] = tstate["fit_itr"]
            btmetrics_d["opt_step"] = tstate["opt_step"]
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(btmetrics_d, "train_dagger"),
                step=tstate["opt_step"],
            )
        # save ckpt if needed
        if ckpt_p is not None and (
            (_itr % save_ckpt_every_n_iter) == 0 or (_itr + 1) == n_iter
        ):
            plf.save(os.path.join(ckpt_p, f"dagger_itr{_itr}.pt"), tstate)
        # rollout with validation set to test performance if needed
        if vdata is not None and (
            _itr % eval_every_n_iter == 0 or (_itr + 1) == n_iter
        ):
            vclassifier = classifier if vclassifier is None else vclassifier
            vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
                    x, nnet=tstate["nnet"], device=plf.device
                ),
                init_fidx=init_fidx,
                tmpls=tmpls,
                metrics_func=metrics_func,
                plf=plf,
            )
            vmetrics_d["fit_itr"] = tstate["fit_itr"]
            vmetrics_d["opt_step"] = tstate["opt_step"]
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(vmetrics_d, "train_dagger"),
                step=tstate["fit_itr"],
            )
        # update fit counter
        tstate["fit_itr"] = tstate["fit_itr"] + 1
    pbar.close()
    return tstate


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
    tpcomp: thd.TensorDict = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tpcomp.pt"),
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
    nnet: th.nn.Module = hd.utils.instantiate(
        cfg.nnet, in_features=2 * n_covs, out_features=len(tmpls)
    )
    # preapre fitting cost estimator
    tstate = _TrainState(
        rplbuf=thrl_data.ReplayBuffer(
            storage=thrl_data.ListStorage(max_size=cfg.max_rplbuf_size),
            sampler=thrl_data.SamplerWithoutReplacement(),
        ),
        nnet=nnet,
        opt=hd.utils.instantiate(cfg.warmup_nnet_tcfg.opt, params=nnet.parameters()),
        fit_itr=0,
        opt_step=0,
    )
    stdata: thd.TensorDict = _daggerlib.compile_selector_dataset(
        tdata=tdata, tpcomp=tpcomp
    )
    # warmup nnet cost estimator
    warmup_fit_nnet_selector(
        tstate=tstate,
        stdata=stdata,
        init_fidx=mktmpl_cfg.init_fidx,
        lmbda=mktmpl_cfg.lmbda,
        tmpls=tmpls,
        n_iter=cfg.warmup_nnet_tcfg.n_fit_iter,
        bsz=cfg.warmup_nnet_tcfg.bsz,
        plf=plf,
        ckpt_p=ckpt_p,
        save_ckpt_every_n_iter=cfg.warmup_nnet_tcfg.save_ckpt_every_n_iter,
    )
    plf.save(os.path.join(ckpt_p, "warmup_end.pt"), tstate)
    # evaluate validation set performance with nnet after warmup
    metrics_d = tafalib.utils.evaluate(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.multi_output_nnet_cost_est(
            x, nnet=nnet, lmbda=mktmpl_cfg.lmbda, tmpls=tmpls, device=plf.device
        ),
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf,
    )
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_warmup"))
    # use dagger to finetune cost estimator
    tstate["opt"] = hd.utils.instantiate(
        cfg.dagger_nnet_tcfg.opt, params=nnet.parameters()
    )
    dagger_fit_nnet_selector(
        tstate=tstate,
        stdata=stdata,
        classifier=tclassifier,
        init_fidx=mktmpl_cfg.init_fidx,
        lmbda=mktmpl_cfg.lmbda,
        tmpls=tmpls,
        to_reset_nnet_after_dagger_rollout=cfg.dagger_nnet_tcfg.to_reset_nnet_after_dagger_rollout,
        n_iter=cfg.dagger_nnet_tcfg.n_fit_iter,
        n_dagger_rollout=cfg.dagger_nnet_tcfg.n_dagger_rollout,
        n_opt_step_per_iter=cfg.dagger_nnet_tcfg.n_opt_step_per_iter,
        bsz=cfg.dagger_nnet_tcfg.bsz,
        metrics_func=metrics_func,
        plf=plf,
        vdata=vdata,
        vclassifier=vclassifier,
        eval_every_n_iter=cfg.dagger_nnet_tcfg.eval_every_n_iter,
        ckpt_p=ckpt_p,
        save_ckpt_every_n_iter=cfg.dagger_nnet_tcfg.save_ckpt_every_n_iter,
    )
    plf.save(os.path.join(ckpt_p, "dagger_end.pt"), tstate)
    # evaluate validation set performance with nnet after dagger finetune
    metrics_d = tafalib.utils.evaluate(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
            x, nnet=nnet, device=plf.device
        ),
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf,
    )
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_dagger"))
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
