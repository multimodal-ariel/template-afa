# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# %%
fig: Figure
axs: list[list[Axes]]
# NOTE ncols dictates how many dataset
fig, axs = plt.subplots(
    nrows=1, ncols=5, squeeze=False, layout="constrained"
)  # type:ignore
color = {0: "blue", 1: "darkorange", 2: "magenta"}
_idx_to_label = {0: "tafa-interp", 1: "sympol", 2: "ddt"}
label_set = list()
line_set = list()
for _i, _ax in enumerate(axs[0]):
    _ax.set_title(f"dataset {_i}")
    _ax.set_box_aspect(1.0)
    for _j in reversed(list(range(3))):
        _ax.plot(
            np.cumsum(np.random.randn(279) * 0.1 + 0.02) * 10,
            np.linspace(0, 1, 279),
            label=_idx_to_label[_j],
            color=color[_j],
        )
    _ax_lines, _ax_labels = _ax.get_legend_handles_labels()
    for _ax_line, _ax_label in zip(_ax_lines, _ax_labels):
        if _ax_label not in label_set:
            line_set.append(_ax_line)
            label_set.append(_ax_label)
# NOTE the \n is needed to make space for figure legend
fig.supxlabel("xlabel\n")
# NOTE use left most ax to set label so taht the label is aligned wrt box
# using fig.supylabel causes label to be aligned wrt the figure
axs[0][0].set_ylabel("ylabel")
fig.legend(
    list(reversed(line_set)),
    list(reversed(label_set)),
    loc="outside lower center",
    frameon=False,
    ncol=6,
    # borderaxespad=1.0,
    # borderaxespad=0.18,
)
fig.set_figheight(2.0 + 0.667 * max(len(label_set) // 4 - 1, 0))
fig.set_figwidth(1.7 * len(axs[0]))
plt.show()

# %%
fig: Figure
axs: list[list[Axes]]
# NOTE ncols dictates how many dataset
fig, axs = plt.subplots(
    nrows=1, ncols=5, squeeze=False, layout="constrained"
)  # type:ignore
color = {0: "blue", 1: "teal", 2: "yellowgreen"}
_idx_to_label = {0: "tafa-interp", 1: "global", 2: "feature-act"}
label_set = list()
line_set = list()
for _i, _ax in enumerate(axs[0]):
    _ax.set_title(f"dataset {_i}")
    _ax.set_box_aspect(1.0)
    for _j in reversed(list(range(3))):
        _ax.plot(
            np.cumsum(np.random.randn(279) * 0.1 + 0.02) * 10,
            np.linspace(0, 1, 279),
            label=_idx_to_label[_j],
            color=color[_j],
        )
    _ax_lines, _ax_labels = _ax.get_legend_handles_labels()
    for _ax_line, _ax_label in zip(_ax_lines, _ax_labels):
        if _ax_label not in label_set:
            line_set.append(_ax_line)
            label_set.append(_ax_label)
# NOTE the \n is needed to make space for figure legend
fig.supxlabel("xlabel\n")
# NOTE use left most ax to set label so taht the label is aligned wrt box
# using fig.supylabel causes label to be aligned wrt the figure
axs[0][0].set_ylabel("ylabel")
fig.legend(
    list(reversed(line_set)),
    list(reversed(label_set)),
    loc="outside lower center",
    frameon=False,
    ncol=6,
    # borderaxespad=1.0,
    # borderaxespad=0.18,
)
fig.set_figheight(2.0 + 0.667 * max(len(label_set) // 4 - 1, 0))
fig.set_figwidth(1.7 * len(axs[0]))
plt.show()

# %%
