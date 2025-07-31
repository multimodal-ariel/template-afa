# %%
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Tuple, TypedDict

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


# %%
@th.no_grad()
def selector_decision_tree_cost_est(inps: th.Tensor, decision_trees) -> th.Tensor:
    """
    Note that this function is only be used to estimate cost of single sample, e.g., inps.shape = (1, n_covs * 2)
    """

    dimensions = inps.shape[1] // 2
    cardinality = inps[dimensions:].sum()

    tree = decision_trees[int(cardinality.item() - 1)]  # since there is init_feature
    probs = tree.predict_proba(inps[0, :].unsqueeze(0).cpu().numpy())[0]

    costs = th.argmin(-probs)
    return costs


# %%
"""
Dzung: things you will have to change (right after this comment): 

- Load your config file here at mktmpl_run_dir
- Load the dir path to the saved checkpoint at model_checkpoint_dir
- add mktmpl_cfg
"""

mktmpl_run_dir: str = "experiments/make_template/outputs/mnist_cnnet/20250326_003820/25"
model_checkpoint_dir: str = (
    "/work/users/d/d/ddinh/template_afa/notebooks/policy/dropout/ensemble/models"
)
mktmpl_cfg = None


"""
the code below is for loading the checkpoint
"""

if "mnist" in mktmpl_run_dir:
    dataset_name = "mnist"
elif "cube" in mktmpl_run_dir:
    dataset_name = "cube_20_0.3"
elif "gas" in mktmpl_run_dir:
    dataset_name = "gas"
elif "grid" in mktmpl_run_dir:
    dataset_name = "grid"
elif "big5" in mktmpl_run_dir:
    dataset_name = "big5_C_cls"


store_num_leafs = {"cube": 4, "mnist": 8, "grid": 8, "gas": 16, "big5": 4}

num_leaf_dataset = store_num_leafs[dataset_name]
teacher_dir = os.path.basename(mktmpl_run_dir)
check_point_dir = (
    f"{dataset_name}_l{num_leaf_dataset}_beta{mktmpl_cfg.lmbda}_accuracy_{teacher_dir}"
)
model_checkpoint_dir = os.path.join(model_checkpoint_dir, check_point_dir)

dt_models = {}
for i, file in enumerate(os.listdir(model_checkpoint_dir)):
    if file.endswith(".joblib"):
        model_path = f"student_{float(i)}.joblib"
        model_path = os.path.join(model_checkpoint_dir, model_path)
        model = joblib.load(model_path)
        dt_models[i] = model


"""
USAGE:
"""
selector_decision_tree_cost_est(inps, dt_models)
