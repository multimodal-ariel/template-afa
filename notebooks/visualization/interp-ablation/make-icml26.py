# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mylib
import pandas as pd

# %%
method_to_plot_kwargs = {
    "ensemble": {
        "label": "tafa-interp",
        "color": "b",
        "alpha": 0.95,
        "marker": ".",
        "linestyle": ":",
        "markersize": 6,
        "linewidth": 3,
    },
    "single_tree": {
        "label": "tafa-interp-non-ensemble",
        "color": "navy",
        "alpha": 0.95,
        "marker": ".",
        # "linestyle": ":",
        "markersize": 6,
        "linewidth": 3,
    },
    "action_feature": {
        "label": "tafa-interp-action-feature",
        "color": "blueviolet",
        "alpha": 0.95,
        "marker": ".",
        # "linestyle": ":",
        "markersize": 6,
        "linewidth": 3,
    },
}


# %%
files_d: dict[str, str] = {
    # "big5": "",
    "blah1": "notebooks/visualization/interp-ablation/ablation/mnist.csv",
    "gas": "notebooks/visualization/interp-ablation/ablation/gas.csv",
    "grid": "notebooks/visualization/interp-ablation/ablation/grid.csv",
    "mnist-ds": "notebooks/visualization/interp-ablation/ablation/mnist.csv",
    # "fashion-ds": "",
    "blah2": "notebooks/visualization/interp-ablation/ablation/mnist.csv",
    # cube
    "blah3": "notebooks/visualization/interp-ablation/ablation/mnist.csv",
}
metrics_d: dict[str, pd.DataFrame] = dict()
for _k, _p in files_d.items():
    metrics_d[_k] = pd.read_csv(os.path.join(mylib.utils.get_project_root_dir(), _p))

# %%
fig: plt.Figure
axs: list[plt.Axes]
fig, axs = plt.subplots(
    nrows=1,
    ncols=len(files_d),
    squeeze=True,
    layout="compressed",
    sharex="col",
)  # type:ignore
label_set = list()
line_set = list()
for (_title, _metrics_df), _ax in zip(metrics_d.items(), axs):
    _ax.set_title(_title, fontsize=23)
    _ax.set_box_aspect(1.0)
    for _method in ["ensemble", "single_tree", "action_feature"]:
        _mdf: pd.DataFrame = _metrics_df.loc[
            _metrics_df["method"] == _method
        ].sort_values(by="num_leaf")
        _ax.plot(
            _mdf["num_leaf"].to_numpy(),
            _mdf["rewards"].to_numpy(),
            **method_to_plot_kwargs[_method],
        )
        _ax_lines, _ax_labels = _ax.get_legend_handles_labels()
        for _ax_line, _ax_label in zip(_ax_lines, _ax_labels):
            if _ax_label not in label_set:
                line_set.append(_ax_line)
                label_set.append(_ax_label)
fig.set_figheight(1.75 * 1.0 + 0.667 * max(len(label_set) // 4 - 1, 0) + 1.0)
fig.set_figwidth(1.85 * len(axs))
fig.supxlabel("number of leaves", fontsize=19)
fig.supylabel("reward", fontsize=19)
fig.legend(
    line_set,
    label_set,
    loc="outside lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.2),
    # ncol=10,
    # ncol=6,
    ncol=3,
    fontsize=17,
    # borderaxespad=1.0,
    # borderaxespad=0.18,
)
fig.savefig("interp-ablation.png", dpi=720, bbox_inches="tight")

# %%
