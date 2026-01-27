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
def plot_stacked_barchart(
    ax: Axes,
    data: pd.DataFrame,
    title: str = "Template Sources Over Rounds",
) -> None:
    """Create stacked bar chart showing template sources over rounds.

    Args:
        ax: Matplotlib axes object to plot on
        data: Processed DataFrame with template source data
        title: Chart title
    """
    # Define colors for each template source
    colors: dict[str, str] = {
        "random": "#3498db",  # Light blue
        "previous": "#f39c12",  # Orange
        "mutated": "#2ecc71",  # Green
    }

    # Extract data arrays
    rounds: Any = data["round"].values
    random_counts: Any = data["random"].values
    previous_counts: Any = data["previous"].values
    mutated_counts: Any = data["mutated"].values

    # Create stacked bars
    width: float = 0.9

    # Bottom layer (random)
    bars1 = ax.bar(rounds, random_counts, width, label="Random", color=colors["random"])

    # Middle layer (previous) - stacked on random
    bars2 = ax.bar(
        rounds,
        previous_counts,
        width,
        bottom=random_counts,
        label="Previous",
        color=colors["previous"],
    )

    # Top layer (mutated) - stacked on random + previous
    bars3 = ax.bar(
        rounds,
        mutated_counts,
        width,
        bottom=random_counts + previous_counts,
        label="Mutated",
        color=colors["mutated"],
    )

    # Customize the chart
    ax.set_xlabel("round", fontsize=16)
    ax.set_ylabel("num. of templates", fontsize=16)
    ax.set_title(title, fontsize=20)

    # Set x-axis to show all rounds
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"{r}" for r in rounds])

    # Set tick label font sizes
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Add legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.96),
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=16,
    )

    # Add grid for better readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


# %%
def plot_validation_accuracy(
    ax: Axes,
    data: pd.DataFrame,
    title: str = "Validation Accuracy Over Rounds",
) -> None:
    """Create line plot showing validation accuracy over rounds.

    Args:
        ax: Matplotlib axes object to plot on
        data: Processed DataFrame with validation accuracy data
        title: Chart title
    """
    # Extract data arrays
    rounds: Any = data["round"].values
    val_acc: Any = data["val_acc"].values

    # Create line plot
    line = ax.plot(
        rounds,
        val_acc,
        color="#e74c3c",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#e74c3c",
        markeredgewidth=2,
        # label="accuracy",
    )

    # Customize the chart
    ax.set_xlabel("round", fontsize=16)
    ax.set_ylabel("accuracy", fontsize=16)
    ax.set_title(title, fontsize=20)

    # Set x-axis to show all rounds
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"{r}" for r in rounds])

    # Format y-axis as decimal
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.3f}"))

    # Set tick label font sizes
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    # ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=False, fontsize=16)


# %%
def add_value_labels_on_bars(
    ax: Axes,
    data: pd.DataFrame,
    show_totals: bool = True,
    show_individual: bool = False,
) -> None:
    """Add value labels on top of bars.

    Args:
        ax: Matplotlib axes object
        data: DataFrame with template source data
        show_totals: Whether to show total values on top of each bar
        show_individual: Whether to show individual segment values
    """
    rounds: Any = data["round"].values

    if show_totals:
        # Add total values on top of each bar
        totals: Any = (data["random"] + data["previous"] + data["mutated"]).values
        for i, (round_num, total) in enumerate(zip(rounds, totals)):
            ax.text(
                round_num,
                total + 1,
                str(int(total)),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

    if show_individual:
        # Add individual segment values (more complex, optional)
        random_counts: Any = data["random"].values
        previous_counts: Any = data["previous"].values
        mutated_counts: Any = data["mutated"].values

        for i, round_num in enumerate(rounds):
            # Random segment (bottom)
            if random_counts[i] > 0:
                ax.text(
                    round_num,
                    random_counts[i] / 2,
                    str(int(random_counts[i])),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white",
                    fontweight="bold",
                    fontfamily="monospace",
                )

            # Previous segment (middle)
            if previous_counts[i] > 0:
                y_pos = random_counts[i] + previous_counts[i] / 2
                ax.text(
                    round_num,
                    y_pos,
                    str(int(previous_counts[i])),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white",
                    fontweight="bold",
                    fontfamily="monospace",
                )

            # Mutated segment (top)
            if mutated_counts[i] > 0:
                y_pos = random_counts[i] + previous_counts[i] + mutated_counts[i] / 2
                ax.text(
                    round_num,
                    y_pos,
                    str(int(mutated_counts[i])),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white",
                    fontweight="bold",
                    fontfamily="monospace",
                )


# %%
def print_summary_statistics(data: pd.DataFrame) -> None:
    """Print summary statistics about template sources and validation accuracy.

    Args:
        data: Processed DataFrame with template source data and validation accuracy
    """
    print("=== Template Source Summary Statistics ===")
    print(f"Total rounds: {len(data)}")
    print(f"Average templates per round: {data['total_selected'].mean():.1f}")
    if "val_acc" in data.columns:
        print(f"Final validation accuracy: {data['val_acc'].iloc[-1]:.3f}")
    print()

    # Calculate totals and percentages
    total_random: int = int(data["random"].sum())
    total_previous: int = int(data["previous"].sum())
    total_mutated: int = int(data["mutated"].sum())
    grand_total: int = total_random + total_previous + total_mutated

    print("Overall composition:")
    print(f"  Random:   {total_random:4d} ({total_random/grand_total*100:5.1f}%)")
    print(f"  Previous: {total_previous:4d} ({total_previous/grand_total*100:5.1f}%)")
    print(f"  Mutated:  {total_mutated:4d} ({total_mutated/grand_total*100:5.1f}%)")
    print(f"  Total:    {grand_total:4d} (100.0%)")
    print()

    # Show round-by-round breakdown
    print("Round-by-round breakdown:")
    for _, row in data.iterrows():
        round_num: int = int(row["round"])
        random_count: int = int(row["random"])
        previous_count: int = int(row["previous"])
        mutated_count: int = int(row["mutated"])
        total: int = random_count + previous_count + mutated_count

        if "val_acc" in data.columns:
            val_acc: float = row["val_acc"]
            print(
                f"  Round {round_num:2d}: R={random_count:3d}, P={previous_count:3d}, M={mutated_count:3d} (Total: {total:3d}), Val Acc: {val_acc:.3f}"
            )
        else:
            print(
                f"  Round {round_num:2d}: R={random_count:3d}, P={previous_count:3d}, M={mutated_count:3d} (Total: {total:3d})"
            )


# %%
run_p: str = (
    "notebooks/visualization/selection-rate/outputs/run/big5_C_cls/mutate-then-select/version_2"
)
run_p = os.path.join(mylib.utils.get_project_root_dir(), run_p)

# %%
metrics_p: str = os.path.join(run_p, "metrics.csv")
metrics_df: pd.DataFrame = (
    pd.read_csv(metrics_p)[
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
# Process the data
processed_data: pd.DataFrame = process_template_source_data(metrics_df)

# Print summary statistics
print_summary_statistics(processed_data)

# Create and save standalone stacked bar chart
fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 6))
plot_stacked_barchart(ax1, processed_data, title="template source")
add_value_labels_on_bars(ax1, processed_data, show_totals=False, show_individual=True)
ax1.set_box_aspect(0.5)

# Save the stacked bar chart
output_path1: str = os.path.join(run_p, "template_sources_stacked_barchart.png")
plt.tight_layout()
plt.savefig(output_path1, dpi=720, bbox_inches="tight")
print(f"Stacked bar chart saved to: {output_path1}")
plt.close()

# Create and save standalone validation accuracy plot
fig2, ax2 = plt.subplots(1, 1, figsize=(3.8, 3.8))
plot_validation_accuracy(ax2, processed_data, title="hold-out accuracy")
ax2.set_box_aspect(1.0)

# Save the validation accuracy plot
output_path2: str = os.path.join(run_p, "validation_accuracy_lineplot.png")
plt.tight_layout()
plt.savefig(output_path2, dpi=720, bbox_inches="tight")
print(f"Validation accuracy plot saved to: {output_path2}")
plt.close()

# %%
