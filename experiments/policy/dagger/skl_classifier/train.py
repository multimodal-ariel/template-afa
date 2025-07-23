from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

import _utils
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import numpy as np
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
    sklc: Any
    n_iter: int
    save_ckpt_every_n_iter: int
    plf: Any


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


class SKLClassifierPolicy:
    def __init__(self, sklc, n_covs: int):
        self.sklc, self.n_covs = sklc, n_covs

    def act(self, state_vec, training=False, epsilon=0.1) -> int:

        probs = self.sklc.predict_proba(state_vec.unsqueeze(0).cpu().numpy())[0]

        best = int(np.argmax(probs))
        return best


def _get_mktmpl_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "mktmpl_run") and cfg.mktmpl_run is not None:
        return cfg.mktmpl_run
    assert hasattr(cfg, "mktmpl_exp") and cfg.mktmpl_exp is not None
    return os.path.join(cfg.mktmpl_exp.exp_p, str(cfg.mktmpl_exp.run_id))


def _dagger_fit(
    tdata: thd.TensorDict,
    tvdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    teacher_cost_est: Callable[[th.Tensor], th.Tensor],
    make_classifier_fn: Callable[[], mymodels.protocols.SklModule],
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    n_iter: int,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
    ckpt_p: Optional[str] = None,
    save_ckpt_every_n_iter: int = 1,
) -> dict[str, SKLClassifierPolicy]:
    student = dict()
    best_student = None
    buffer_state = {f"{float(idx)}": [] for idx in range(tmpls.shape[1])}
    buffer_action = {f"{float(idx)}": [] for idx in range(tmpls.shape[1])}
    buffer_weight = {f"{float(idx)}": [] for idx in range(tmpls.shape[1])}
    track_obj = float("inf")
    pbar = tqdm.trange(n_iter, desc="dagger-sklc", leave=True, dynamic_ncols=True)
    for _itr in pbar:
        # expert demonstration
        metrics_func.reset()
        metrics_d, buffer_state, buffer_action, buffer_weight = _utils.evaluate(
            data=tdata,
            classifier=classifier,
            cost_est=teacher_cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            metrics_func=metrics_func,
            plf=plf,
            student=student,
            training=True,
            buffer_state=buffer_state,
            buffer_action=buffer_action,
            buffer_weight=buffer_weight,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "expert_demo"), step=_itr
        )
        # student policy training
        student = dict()  # do we need this?
        for _cardinality in buffer_state.keys():
            if not buffer_state[_cardinality]:
                continue
            actions = np.array(buffer_action[_cardinality])
            states = np.concatenate(buffer_state[_cardinality], axis=0)
            weights = np.array(buffer_weight[_cardinality])
            if states.shape[0] == 0:
                continue
            _single_sklc = make_classifier_fn()
            _all_classes = set(range(tmpls.shape[0]))
            _present = set(actions)
            _missing = _all_classes - _present
            if _missing:
                _dummy_x = np.zeros((1, states.shape[1]), dtype=states.dtype)
                for a in sorted(_missing):
                    states = np.vstack([states, _dummy_x])
                    actions = np.append(actions, a)
                    weights = np.append(weights, 1e-6)
            _single_sklc.fit(states, actions)  # , sample_weight=weights
            _single_student = SKLClassifierPolicy(_single_sklc, tmpls.shape[1])
            student[f"{_cardinality}"] = _single_student
        # cross validation with 3rd holdout train set
        metrics_func.reset()
        metrics_d, _, _, _obj = _utils.evaluate(
            data=tvdata,
            classifier=classifier,
            cost_est=teacher_cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            metrics_func=metrics_func,
            plf=plf,
            student=student,
            training=False,
            lmbda=lmbda,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "student_clone"), step=_itr
        )
        if _obj < track_obj:
            track_obj = _obj
            best_student = student
            pbar.set_postfix(
                {
                    "best": _obj,
                    "_itr": _itr,
                }
            )
            # print(
            #     f"*** New best student found with obj {track_obj:.4f} at iteration {_itr + 1}"
            # )
        # save ckpt if needed
        if ckpt_p is not None and (
            (_itr % save_ckpt_every_n_iter) == 0 or (_itr + 1) == n_iter
        ):
            plf.save(
                os.path.join(ckpt_p, f"dagger_itr{_itr}.pt"),
                {
                    "best_student": best_student,
                    "student": student,
                },
            )
    pbar.close()
    assert best_student is not None
    return best_student


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
    os.makedirs(ckpt_p, exist_ok=True)
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
    # dagger fit
    best_student: dict[str, SKLClassifierPolicy] = _dagger_fit(
        tdata=tdata,
        tvdata=tstdata,
        classifier=tclassifier,
        teacher_cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=mktmpl_cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=mktmpl_cfg.n_neighs,
            p=2,
        ),
        make_classifier_fn=hd.utils.instantiate(cfg.sklc, partial=True),
        init_fidx=mktmpl_cfg.init_fidx,
        lmbda=mktmpl_cfg.lmbda,
        tmpls=tmpls,
        n_iter=cfg.n_iter,
        metrics_func=metrics_func,
        plf=plf,
        ckpt_p=ckpt_p,
        save_ckpt_every_n_iter=cfg.save_ckpt_every_n_iter,
    )
    # evaluate validation set performance
    metrics_d = _utils.evaluate(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=mktmpl_cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=mktmpl_cfg.n_neighs,
            p=2,
        ),
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf,
        student=best_student,
        training=False,
        lmbda=mktmpl_cfg.lmbda,
    )[0]
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
