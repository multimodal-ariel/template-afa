# %%
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mylib
import pandas as pd
from matplotlib.axes import Axes

# %%
greedy_p: str = (
    "notebooks/visualization/selection-rate/outputs/run/big5_C_cls/greedy/version_3"
)
greedy_p = os.path.join(mylib.utils.get_project_root_dir(), greedy_p)
mutate_p: str = (
    "notebooks/visualization/selection-rate/outputs/run/big5_C_cls/mutate/version_2"
)
mutate_p = os.path.join(mylib.utils.get_project_root_dir(), mutate_p)
greedy_df: pd.DataFrame = (
    pd.read_csv(os.path.join(greedy_p, "metrics.csv"))[["eval-val/acc", "step"]]
    .groupby("step")
    .sum(min_count=1)
    .dropna(ignore_index=True)
)
mutate_df: pd.DataFrame = (
    pd.read_csv(os.path.join(mutate_p, "metrics.csv"))[
        [
            "round-stats/template-source-mutated",
            "round-stats/template-source-previous",
            "round-stats/template-source-random",
            "round-stats/total_selected",
            "round",
            "val/acc",
        ]
    ]
    .groupby("round")
    .sum(min_count=1)
    .dropna(ignore_index=True)
)


# %%
def process_template_source_data(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Process metrics DataFrame to prepare data for visualization.

    Args:
        metrics_df: DataFrame containing template source columns and validation accuracy

    Returns:
        Processed DataFrame with round numbers and clean column names
    """
    # Create a copy and add round numbers
    processed_df: pd.DataFrame = metrics_df.copy()
    processed_df["round"] = range(len(processed_df))

    # Rename columns for easier access
    processed_df = processed_df.rename(
        columns={
            "round-stats/template-source-mutated": "mutated",
            "round-stats/template-source-previous": "previous",
            "round-stats/template-source-random": "random",
            "round-stats/total_selected": "total_selected",
            "val/acc": "val_acc",
        }
    )

    return processed_df


# %%
