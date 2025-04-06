# %%
from __future__ import annotations

import os
import shutil
import logging

import mylib
import pandas as pd
import tensordict as thd
import torch as th
import itertools as itrtls

# %%
# exp_p: str = "experiments/baselines/aco/outputs/mnist/20250323_183552"
exp_p: str = "experiments/baselines/aco/outputs/mnist/20250312_143945"

# %%
lgr = logging.getLogger()

# %%
for run_id in itrtls.count():
    # walk through each run directory
    run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, str(run_id))
    if not os.path.exists(run_p):
        break
    # load saved experiment results
    results_p: str = os.path.join(run_p, "results.pt")
    if not os.path.exists(results_p):
        lgr.warning(f"{results_p} not exists")
        continue
    metrics_p: str = os.path.join(run_p, "metrics.csv")
    if not os.path.exists(metrics_p):
        lgr.warning(f"{metrics_p} not exists")
        continue
    results: thd.TensorDict = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), results_p), weights_only=False
    )
    metrics_df: pd.DataFrame = (
        pd.read_csv(os.path.join(mylib.utils.get_project_root_dir(), metrics_p))
        .groupby("step")
        .sum(min_count=1)
    )
    # # leave unchanged if feature observed is already included
    # if "eval_val/feature observed" in metrics_df:
    #     continue
    # compute average features used
    metrics_df["eval_val/feature observed"] = th.mean(
        th.sum(
            results["mask"][th.argwhere(results["Action"][:, -1] == 1).flatten(), :],
            dim=1,
        )
    ).item()
    # backup old metrics first
    shutil.copyfile(metrics_p, f"{metrics_p}.bak")
    # overwrite
    metrics_df.to_csv(metrics_p)

# %%
