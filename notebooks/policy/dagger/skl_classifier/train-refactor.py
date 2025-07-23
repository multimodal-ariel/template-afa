"""
Dzung: need to change max_leaf_nodes for each dataset.
"""

# %%

from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Callable, Optional

import _utils
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import numpy as np
import pandas as pd
import sklearn.base as skl_base
import sklearn.tree as skl_tree
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = mylib.utils.get_project_root_dir()
# PROJ_ROOT_HT = "/playpen1/scribble/hungtien/template-afa"
# print("*** PROJ_ROOT:", PROJ_ROOT)

# %%
mktmpl_run_dir: str = (
    "experiments/make_template/outputs/big5_cnnet/20250318_144121/5"  # big5
)

# %%
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tdata_shuffle_idxs.pt")
)
tmpls: th.Tensor = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tmpls.pt"), weights_only=False
)
tpcomp: thd.TensorDict = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tpcomp.pt"), weights_only=False
)
mktmpl_cfg = OmegaConf.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, ".hydra", "config.yaml")
)

# %%
# make dataset
_tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
_tdata, vdata, tstdata = hd.utils.call(mktmpl_cfg.data)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
# split training data into two for classifier and afa
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
# make classifier
# NOTE cnnet
# import pdb; pdb.set_trace()
tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
    mktmpl_cfg.tclassifier,
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
)
if os.path.exists(
    os.path.join(mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt")
):
    tclassifier.load_state_dict(
        th.load(
            os.path.join(
                mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
            )
        )
    )

if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
    tclassifier_p = os.path.join(
        mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
    )
    if os.path.exists(tclassifier_p):
        tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
    else:
        tclassifier.fit_(tmpls)

# %%
vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
if mktmpl_cfg.vclassifier is not None:
    vclassifier = hd.utils.instantiate(
        mktmpl_cfg.vclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )

# %%
# configure logger and ckpt path
output_dir: str = os.path.join(
    "outputs", "run", mktmpl_cfg.data.name, "dagger-refactor"
)
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")
vclassifier = vclassifier.to(plf.device)

# %%
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)

# %%
print("*** Testing knn policy ***")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
metrics_func.reset()
metrics_d, _, _, _ = _utils.evaluate(
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
    student=None,
    training=True,
)
print(pd.Series(metrics_d))


# %%
class SKLClassifierPolicy:
    def __init__(self, sklc, n_covs: int):
        self.sklc, self.n_covs = sklc, n_covs

    def act(self, state_vec, training=False, epsilon=0.1) -> int:

        probs = self.sklc.predict_proba(state_vec.unsqueeze(0).cpu().numpy())[0]

        best = int(np.argmax(probs))
        return best


def _dagger_fit(
    tdata: thd.TensorDict,
    tvdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    teacher_cost_est: Callable[[th.Tensor], th.Tensor],
    make_classifier_fn: Callable[
        [], mymodels.protocols.SklModule | skl_base.BaseEstimator
    ],
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
                    "best": _obj.item(),
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


# %%
iterations = 10
max_leaf_nodes = 256

# %%
best_student = _dagger_fit(
    tdata=tdata,
    tvdata=extdata,
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
    make_classifier_fn=partial(
        skl_tree.DecisionTreeClassifier, max_leaf_nodes=max_leaf_nodes
    ),
    init_fidx=mktmpl_cfg.init_fidx,
    lmbda=mktmpl_cfg.lmbda,
    tmpls=tmpls,
    n_iter=iterations,
    metrics_func=metrics_func,
    plf=plf,
    ckpt_p=None,
    save_ckpt_every_n_iter=1,
)

# %%
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
print(metrics_d)
plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"))
# logger flush record and close
tfb_logger.finalize("success")
csv_logger.finalize("success")

# %%
