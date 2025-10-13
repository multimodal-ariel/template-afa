# %%
from __future__ import annotations

import logging as lgr
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import mylib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from omegaconf import OmegaConf

# %%
exp_p: str = "experiments/make_template/outputs/cube-o_init-sweep/20251009_070624"


# %%
def safe_str_to_int(s: str):
    try:
        return int(s)
    except ValueError as e:
        return -1


def load_tafa_metrics(
    exp_p: str, exclude_mktmplfn_name: Optional[list[str]]
) -> dict[int, dict[str, list[pd.DataFrame]]]:
    """Modified to include lambda tracking"""
    metrics_d: dict[int, dict[str, list[pd.DataFrame]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue

        run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
        mktmpl_fn_name: str = str.split(run_cfg.make_templates_fn._target_, ".")[-1]

        if (
            exclude_mktmplfn_name is not None
            and mktmpl_fn_name in exclude_mktmplfn_name
        ):
            continue

        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )

        if (
            "eval_val/feature observed" not in metrics_df
            and "eval/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue

        # Add lambda value to dataframe
        lmbda_val = getattr(run_cfg, "lmbda", None)
        if lmbda_val is not None:
            metrics_df["lmbda"] = lmbda_val

        metrics_d[run_cfg.init_fidx][mktmpl_fn_name].append(metrics_df)
    return metrics_d


# %%
metrics_dfs_d = load_tafa_metrics(exp_p=exp_p, exclude_mktmplfn_name=None)


# %%
def create_lambda_cost_comparison(metrics_dfs_d):
    """Create subplots comparing eval/reward vs initial feature index for each lambda value"""
    # Collect all data points with their lambda values, init_fidx, and rewards
    data_points = []
    for init_fidx, methods_dict in metrics_dfs_d.items():
        for method_name, df_list in methods_dict.items():
            for df in df_list:
                if "lmbda" in df.columns and "eval/reward" in df.columns:
                    # Get the last step's data (assuming we want final rewards)
                    last_row = df.iloc[-1]
                    data_points.append(
                        {
                            "init_fidx": init_fidx,
                            "lmbda": last_row["lmbda"],
                            "cost": -last_row["eval/reward"],
                            "method": method_name,
                        }
                    )
    # Convert to DataFrame for easier manipulation
    plot_df = pd.DataFrame(data_points)
    # Get unique lambda values
    unique_lambdas = sorted(plot_df["lmbda"].unique())
    # Calculate subplot layout
    n_lambdas = len(unique_lambdas)
    # n_cols = min(4, n_lambdas)  # Max 4 columns
    n_cols = 3
    n_rows = (n_lambdas + n_cols - 1) // n_cols
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    # Handle case where we have only one subplot
    if n_lambdas == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    # Create bar chart for each lambda value
    for i, lmbda in enumerate(unique_lambdas):
        ax: Axes = axes[i]
        # Filter data for this lambda value
        lambda_data = plot_df[plot_df["lmbda"] == lmbda]
        # Group by init_fidx and calculate mean reward (in case of multiple runs)
        grouped_data = (
            lambda_data.groupby("init_fidx")["cost"].agg(["mean", "std"]).reset_index()
        )
        # Create bar chart
        bars = ax.bar(
            grouped_data["init_fidx"],
            grouped_data["mean"],
            yerr=grouped_data["std"],
            capsize=3,
            alpha=0.7,
        )
        ax.set_xlabel("Initial Feature Index")
        ax.set_ylabel("Eval/cost")
        ax.set_title(f"λ = {lmbda}")
        ax.set_ylim(1.5, 2.3)
        ax.grid(True, alpha=0.3)
        # ax.set_aspect(1.0)
        # Set x-axis to show all init_fidx values
        ax.set_xticks(grouped_data["init_fidx"])
    # Hide unused subplots
    for i in range(n_lambdas, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig, plot_df


# %%
create_lambda_cost_comparison(metrics_dfs_d)


# %%
def plot_single_lambda_cost(metrics_dfs_d, target_lambda, figsize=(8, 6)):
    """Create a single bar chart for a specific lambda value showing eval/reward vs initial feature index"""
    # Collect data points for the target lambda
    data_points = []
    for init_fidx, methods_dict in metrics_dfs_d.items():
        for method_name, df_list in methods_dict.items():
            for df in df_list:
                if "lmbda" in df.columns and "eval/reward" in df.columns:
                    # Get the last step's data
                    last_row = df.iloc[-1]
                    if last_row["lmbda"] == target_lambda:
                        data_points.append(
                            {
                                "init_fidx": init_fidx,
                                "lmbda": last_row["lmbda"],
                                "cost": -last_row["eval/reward"],
                                "method": method_name,
                            }
                        )
    if not data_points:
        print(f"No data found for lambda = {target_lambda}")
        return None, None
    # Convert to DataFrame
    plot_df = pd.DataFrame(data_points)
    # Group by init_fidx and calculate mean reward (in case of multiple runs)
    grouped_data = (
        plot_df.groupby("init_fidx")["cost"].agg(["mean", "std"]).reset_index()
    )
    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)
    # Create bar chart
    bars = ax.bar(
        grouped_data["init_fidx"],
        grouped_data["mean"],
        yerr=grouped_data["std"],
        capsize=5,
        alpha=0.7,
        color="steelblue",
    )
    ax.set_xlabel("Initial Feature Index")
    ax.set_ylabel("Eval/cost")
    ax.set_title(f"Eval/cost vs Initial Feature Index (λ = {target_lambda})")
    ax.set_ylim(2.0, 2.3)
    # ax.set_aspect(1.0)
    ax.grid(True, alpha=0.3)
    # Set x-axis to show all init_fidx values
    ax.set_xticks(grouped_data["init_fidx"])
    # Add value labels on top of bars
    for bar, mean_val in zip(bars, grouped_data["mean"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()
    return fig, ax, plot_df


# %%
plot_single_lambda_cost(metrics_dfs_d, target_lambda=0.12)

# %%
