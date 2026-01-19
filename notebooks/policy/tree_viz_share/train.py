# %%
"""
Dzung: This code is ready for the preliminary experiments. Note that the same lambda for knn and xgb policy will not result in the same cost.

In the future, we will need to tune:
- lambda
- n_pairs_per_sample
- n_estimators, max_depth, etc. for xgb
"""
# %%
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Callable, Optional, Tuple, TypedDict

import _utils
import hydra as hd
import joblib
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import numpy as np
import pandas as pd
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
from omegaconf import OmegaConf
from sklearn.tree import DecisionTreeClassifier
from torch.nn import functional as F
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="To copy construct from a tensor.*"
)


# %%
# @hydra.main(version_base="1.3", config_path=None, config_name=None)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a decision‑tree model given a dataset."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Path to your dataset file (e.g. CSV or NPZ).",
    )
    parser.add_argument(
        "--max_leaf_nodes",
        "-m",
        type=int,
        required=True,
        help=(
            "The maximum number of leaf nodes in the decision tree. "
            "If not set, the tree will expand until purity or min samples per leaf."
        ),
    )
    # the dir for mktmpl_run_dir
    parser.add_argument(
        "--mktmpl_run_dir",
        "-r",
        type=str,
        default="experiments/make_template/outputs/mnist_cnnet/20250326_003820/6",
        help=(
            "The directory where the make template run is stored. "
            "This is used to load the templates and other necessary data."
        ),
    )

    return parser.parse_args()


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = mylib.utils.get_project_root_dir()
PROJ_ROOT_HT = "/playpen1/scribble/hungtien/template-afa"
print("*** PROJ_ROOT:", PROJ_ROOT)

args = parse_args()

_dataset_name = args.dataset
max_leaf_nodes = args.max_leaf_nodes


# mktmpl_run_dir: str = "/playpen1/ddinh/interpretable_afa/template-afa/experiments/make_template/outputs/gas_cnnet/20250324_224734/0"
# mktmpl_run_dir: str = "experiments/make_template/outputs/cube/20250305_143844/4"
# mktmpl_run_dir: str = "experiments/make_template/outputs/big5_cnnet/20250314_112541/14"
# mktmpl_run_dir: str = "/playpen1/ddinh/interpretable_afa/template-afa/experiments/make_template/outputs/cube/0" # cube
mktmpl_run_dir = args.mktmpl_run_dir
import os

last_part = os.path.basename(mktmpl_run_dir)

# big5: 25= 0.18, 3 = 0.06, 5 = 0.1, 4=0.08
# mnist:  6 = 0.02, 10 = 0.03, 18 = 0.05, 26 = 0.07

# big5 = 0.1, mnist 0.05, cube = 0.06


@th.no_grad()
def compile_selector_dataset(
    tdata: thd.TensorDict, tpcomp: thd.TensorDict
) -> thd.TensorDict:
    # (n_data, n_covs)
    xs: th.Tensor = tdata["xs"]
    # (n_data, )
    ys: th.Tensor = tdata["ys"]
    # (n_data, n_tmpls, n_labels)
    pyhats: th.Tensor = tpcomp["pyhats"]
    # (n_data, n_tmpls)
    cels: th.Tensor = tpcomp["cels"]
    rwds: th.Tensor = tpcomp["rwds"]
    # # (n_data, n_tmpls)
    # slbls: th.Tensor = th.softmax(rwds / tau_rwd, dim=1)
    # bundle tensors into tensordict
    stdata = thd.TensorDict(
        {
            "xs": xs,
            "ys": ys,
            "pyhats": pyhats,
            "cels": cels,
            "rwds": rwds,
        }
    ).auto_batch_size_(1)
    return stdata


# Dzung: below are my ultility functions
@th.no_grad()
def _compute_ce_batch(
    xs_u: np.ndarray,  # (B, n_covs)
    ms_u: np.ndarray,  # (B, n_covs)
    ys: np.ndarray,  # (B,)
    classifier: th.nn.Module,
    device: th.device,
) -> np.ndarray:
    B, n_covs = xs_u.shape
    xs_t = th.tensor(xs_u, dtype=th.float32, device=device)
    ms_t = th.tensor(ms_u, dtype=th.float32, device=device)
    ys_t = th.tensor(ys, dtype=th.long, device=device)

    probs = classifier.to(device).predict_proba(xs_t, ms_t)
    probs = probs.to(device=device, dtype=th.float64)
    ce = F.nll_loss(probs.log(), ys_t, reduction="none")
    return ce.cpu().numpy().astype(np.float32)

    # ce = -th.log(probs[th.arange(B), ys_t] + 1e-12)
    # return ce.cpu().numpy().astype(np.float32)


def make_xgb_ce_dataset(
    xs: th.Tensor,  # (n_data, n_covs)
    ys: th.Tensor,  # (n_data,)
    tmpls: th.Tensor,  # (n_tmpls, n_covs)
    classifier: th.nn.Module,
    init_fidx: int,
    *,
    n_pairs_per_sample: int = 128,
    rng: np.random.Generator = np.random.default_rng(0),
    device: th.device = th.device("cuda:0"),
):
    xs_np, ys_np, tmpls_np = xs.numpy(), ys.numpy(), tmpls.numpy()
    n_data, n_covs = xs_np.shape
    X_rows, y_rows = [], []

    for i in range(n_data):
        x_i, y_i = xs_np[i], ys_np[i]

        idx_m1 = rng.integers(len(tmpls_np), size=n_pairs_per_sample)
        idx_m2 = rng.integers(len(tmpls_np), size=n_pairs_per_sample)
        m1_all = tmpls_np[idx_m1].copy()
        m2_all = tmpls_np[idx_m2]

        # randomly drop some features in m1
        m1_all -= rng.integers(0, 2, size=m1_all.shape)
        np.clip(m1_all, 0, 1, out=m1_all)
        m1_all[:, init_fidx] = 1

        m_union = np.maximum(m1_all, m2_all)
        ce_vals = _compute_ce_batch(
            xs_u=np.broadcast_to(x_i, m_union.shape),
            ms_u=m_union,
            ys=np.full(n_pairs_per_sample, y_i),
            classifier=classifier,
            device=device,
        )

        x_obs = x_i * m1_all
        rows = np.concatenate([x_obs, m1_all, m2_all], axis=1)  # (k, 3 * n_covs)

        X_rows.append(rows)
        y_rows.append(ce_vals)

    X = np.vstack(X_rows).astype(np.float32)
    y = np.hstack(y_rows).astype(np.float32)
    return X, y


@th.no_grad()
def xgb_union_ce_batch(
    ctx: th.Tensor,
    *,
    tmpls: th.Tensor,
    xgb_model: XGBRegressor,
    device: th.device,
    rows_per_batch: int = 200000,  # not using this right now
) -> th.Tensor:
    B, two_nc = ctx.shape
    n_covs = two_nc // 2
    x_obs_np = ctx[:, :n_covs].cpu().numpy()
    m1_np = ctx[:, n_covs:].cpu().numpy()
    tmpls_np = tmpls.cpu().numpy()
    n_tmpls = tmpls_np.shape[0]

    x_rep = np.repeat(x_obs_np, n_tmpls, axis=0)
    m1_rep = np.repeat(m1_np, n_tmpls, axis=0)
    m2_rep = np.tile(tmpls_np, (B, 1))
    X_full = np.concatenate([x_rep, m1_rep, m2_rep], axis=1).astype(np.float32)

    # preds = np.empty(X_full.shape[0], dtype=np.float32)
    # for start in range(0, X_full.shape[0], rows_per_batch):
    #     end = min(start + rows_per_batch, X_full.shape[0])
    #     preds[start:end] = xgb_model.predict(X_full[start:end])

    # X_full = cp.asarray(X_full)       # move data to GPU
    preds = xgb_model.predict(X_full)

    return th.tensor(preds.reshape(B, n_tmpls), device=device)


@th.no_grad()
def xgb_cost_est(
    inps: th.Tensor,  # (B, 2 * n_covs)
    *,
    lmbda: float,
    tmpls: th.Tensor,
    xgb_model: XGBRegressor,
    device: th.device = th.device("cpu"),
    rows_per_batch: int = 200000,
) -> th.Tensor:

    # TODO: scale tafalib.utils.evaluate so that it can handle inputs in batches

    ce_pred = xgb_union_ce_batch(
        inps,
        tmpls=tmpls,
        xgb_model=xgb_model,
        device=device,
        rows_per_batch=rows_per_batch,  # (B, n_tmpls)
    )

    # acq cost
    B, two_nc = inps.shape
    n_covs = two_nc // 2
    m1 = inps[:, n_covs:].to(device)  # (B, n_covs)
    tmpls_exp = tmpls.to(device)[None, :, :]  # (1, n_tmpls, n_covs)

    to_acq = th.clamp(tmpls_exp - m1[:, None, :], 0, 1)
    acq_cost = lmbda * to_acq.sum(dim=2)  # (B, n_tmpls)

    return ce_pred + acq_cost


class DecisionTreePolicy:
    def __init__(self, tree, d: int):
        self.tree, self.d = tree, d

    def act(self, state_vec, training=False, epsilon=0.1):

        probs = self.tree.predict_proba(state_vec.unsqueeze(0).cpu().numpy())[0]

        best = int(np.argmax(probs))

        return best


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
            ),
            map_location="cpu",
        )
    )
# NOTE xgboost
# tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
#     cfg.tclassifier, xs_train=extdata["xs"].numpy(), ys_train=extdata["ys"].numpy()
# )
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
stdata: thd.TensorDict = compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", mktmpl_cfg.data.name, "cnnet_mixin")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

vclassifier = vclassifier.to(plf.device)

import pickle

with open(
    "/work/users/d/d/ddinh/template_afa/notebooks/policy/dropout/ensemble/cube.pkl",
    "wb",
) as f:
    pickle.dump(vclassifier, f)

print("*** Running dir:", mktmpl_run_dir)
print(
    "*** Running with dataset:",
    _dataset_name,
    " lambda:",
    mktmpl_cfg.lmbda,
    " num_leaves:",
    max_leaf_nodes,
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

student = {}
best_student = None

iterations = 5
top_k = 1


beta_0 = 1
gamma = 0.85
beta_min = 0.1

buffer_state = {f"{float(idx)}": [] for idx in range(tmpls.shape[1])}
buffer_action = {f"{float(idx)}": [] for idx in range(tmpls.shape[1])}
buffer_weight = {f"{float(idx)}": [] for idx in range(tmpls.shape[1])}
track_obj = float("inf")
track_size = {f"{float(idx)}": 0 for idx in range(tmpls.shape[1])}

# create a csv file that will be used to log the results
log_file_dir = (
    "/work/users/d/d/ddinh/template_afa/notebooks/policy/dropout/ensemble/results"
)
os.makedirs(log_file_dir, exist_ok=True)
log_file_path = os.path.join(
    log_file_dir,
    f"{_dataset_name}_l{max_leaf_nodes}_beta{mktmpl_cfg.lmbda}_accuracy_{last_part}.csv",
)
log_file = open(log_file_path, "w")
log_file.write("iteration,obj,acc,auroc,f1-score,precision,recall,features\n")

for it in range(iterations):
    # TRAINING

    # epsilon_dagger = 0
    epsilon_dagger = max(beta_min, beta_0 * (gamma**it))

    print("*** Iteration", it + 1, "/", iterations, "***")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    metrics_func.reset()
    metrics_d, buffer_state, buffer_action, buffer_weight = _utils.evaluate(
        data=tdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=mktmpl_cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=mktmpl_cfg.n_neighs,
            p=2,
            is_train=True,
        ),
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf,
        student=student,
        training=True,
        top_k=top_k,
        buffer_state=buffer_state,
        buffer_action=buffer_action,
        buffer_weight=buffer_weight,
        epsilon=epsilon_dagger,
    )

    student = {}  # do we need this?

    for cardinality in buffer_state.keys():
        if not buffer_state[cardinality]:
            continue
        # states = np.array(buffer_state[cardinality])
        actions = np.array(buffer_action[cardinality])
        states = np.concatenate(buffer_state[cardinality], axis=0)
        weights = np.array(buffer_weight[cardinality])

        if states.shape[0] == 0:
            continue

        n_total = states.shape[0]
        n_new_rows = n_total - track_size[cardinality]

        if n_new_rows > 0:
            w_new = np.full(n_new_rows, 1.0 / epsilon_dagger, dtype=np.float32)
            buffer_weight[cardinality].extend(w_new)
            track_size[cardinality] += n_new_rows

        weights = np.asarray(buffer_weight[cardinality], dtype=np.float32)

        single_tree = DecisionTreeClassifier(
            max_leaf_nodes=max_leaf_nodes,
        )

        all_classes = set(range(tmpls.shape[0]))
        present = set(actions)
        missing = all_classes - present

        if missing:
            dummy_x = np.zeros((1, states.shape[1]), dtype=states.dtype)
            for a in sorted(missing):
                states = np.vstack([states, dummy_x])
                actions = np.append(actions, a)
                weights = np.append(weights, 1e-6)

        single_tree.fit(
            states,
            actions,  # sample_weight=weights
        )

        single_student = DecisionTreePolicy(single_tree, tmpls.shape[1])

        student[f"{cardinality}"] = single_student

    # EVALUATION
    metrics_func.reset()
    metrics_d, _, _, obj = tafalib.utils.evaluate(
        data=extdata,
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
        student=student,
        training=False,
        lmbda=mktmpl_cfg.lmbda,
    )
    print(pd.Series(metrics_d))

    if obj < track_obj:
        track_obj = obj
        best_student = student

        saved_path = f"/work/users/d/d/ddinh/template_afa/notebooks/policy/dropout/ensemble/models/{mktmpl_cfg.data.name}_l{max_leaf_nodes}_beta{mktmpl_cfg.lmbda}_accuracy_{last_part}"
        os.makedirs(saved_path, exist_ok=True)
        for cardinality, student_policy in student.items():
            student_path = os.path.join(saved_path, f"student_{cardinality}.joblib")
            joblib.dump(student_policy.tree, student_path)

            # print(f"Saved student policy for cardinality {cardinality} to {student_path}")
        # log the results
        #  [feature used]
        log_file.write(
            f"{it + 1},{track_obj:.4f},{metrics_d['acc']:.4f},{metrics_d['auroc']:.4f},{metrics_d['f1-score']:.4f},{metrics_d['precision']:.4f},{metrics_d['recall']:.4f},{metrics_d['feature used']:.4f}\n"
        )
        log_file.flush()
        print(
            f"*** New best student found with obj {track_obj:.4f} at iteration {it + 1}"
        )
        # best_student

    else:
        print(
            f"*** Student at iteration {it + 1} did not improve the best student with new obj {obj:.4f}"
        )

    # TESTING
    metrics_func.reset()
    metrics_d, _, _, obj_test = tafalib.utils.evaluate(
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
        student=student,
        training=False,
        lmbda=mktmpl_cfg.lmbda,
    )

    print(pd.Series(metrics_d))
    print("*** Test obj:", obj_test)

# for leaf in 2 4 8 16; do for idx in {0..0}; do MKTMPL_DIR="/work/users/d/d/ddinh/template_afa/experiments/make_template/outputs/engine-cnnet/20260110_215609/${idx}"; echo "Submitting Leaf=${leaf}, Idx=${idx}"; sbatch --export=ALL,HP_DATASET="enginecnnet",HP_MAX_LEAF="${leaf}",HP_MKTEMPL_RUN_DIR="${MKTMPL_DIR}" /work/users/d/d/ddinh/tafa_accuracy.s1; done; done
# for idx in {0..15}; do MKTMPL_DIR="/work/users/d/d/ddinh/template_afa/experiments/make_template/outputs/cube/20260105_232445/${idx}"; echo "Submitting Leaf=4, Idx=${idx}"; sbatch --export=ALL,HP_DATASET="cube",HP_MAX_LEAF="4",HP_MKTEMPL_RUN_DIR="${MKTMPL_DIR}" /work/users/d/d/ddinh/tafa_accuracy.s1; done
# for leaf in 4 8; do for idx in {0..15}; do MKTMPL_DIR="/work/users/d/d/ddinh/template_afa/experiments/make_template/outputs/big5_cnnet/20251213_024429/${idx}"; echo "Submitting Leaf=${leaf}, Idx=${idx}"; sbatch --export=ALL,HP_DATASET="big5",HP_MAX_LEAF="${leaf}",HP_MKTEMPL_RUN_DIR="${MKTMPL_DIR}" /work/users/d/d/ddinh/tafa_accuracy.s1; done; done
# for leaf in 8 16; do for idx in {0..14}; do MKTMPL_DIR="/work/users/d/d/ddinh/template_afa/experiments/make_template/outputs/fashion-16x16_cnnet/20260102_202634/${idx}"; echo "Submitting Leaf=${leaf}, Idx=${idx}"; sbatch --export=ALL,HP_DATASET="fashion",HP_MAX_LEAF="${leaf}",HP_MKTEMPL_RUN_DIR="${MKTMPL_DIR}" /work/users/d/d/ddinh/tafa_accuracy.s1; done; done
# for leaf in 8 16 32; do for idx in {11..21}; do MKTMPL_DIR="/work/users/d/d/ddinh/template_afa/experiments/make_template/outputs/fashion_cnnet/20251120_232234/${idx}"; echo "Submitting Leaf=${leaf}, Idx=${idx}"; sbatch --export=ALL,HP_DATASET="fashionfull",HP_MAX_LEAF="${leaf}",HP_MKTEMPL_RUN_DIR="${MKTMPL_DIR}" /work/users/d/d/ddinh/tafa_accuracy.s1; done; done
