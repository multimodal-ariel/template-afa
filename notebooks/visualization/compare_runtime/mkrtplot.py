# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# %%
outputs_p: str = "outputs"
os.makedirs(outputs_p, exist_ok=True)

# %%
# nfeats_aaco = [36.70, 78.900, 6.3, 8.30, 11.30]
# nfeats_tafa = [47.2, 10.89, 6.0, 10.60, 34.79]
fn: str = "time.png"
ts_aaco = [5.9212, 1.2519, 29.280, 29.0495, 26.1010]
ts_tafa = [0.34837, 0.0248, 0.2254, 0.05618, 0.6679]
ts_dime = [0.73881, 0.4489, 0.0976, 0.2670, 5.632]
ts_jafa = [0.00480, 0.0023, 0.0010, 0.0008, 0.00292]
labels_l = ["big5", "cube", "gas", "grid", "mnist"]

# %%
n_bars: int = 4
width = 0.2
ind = np.arange(len(labels_l))
offset = (n_bars * width) / 2

# %%
fig: Figure
ax: Axes
fig, ax = plt.subplots(layout="constrained")
# fig.tight_layout()
fig.set_figheight(2.0)
fig.set_figwidth(0.35 * 5 + len(labels_l) * 0.4)
ax.bar(ind - offset + 0 * width, ts_aaco, width, label="aco", color="red")
ax.bar(ind - offset + 1 * width, ts_tafa, width, label="ours", color="blue")
ax.bar(ind - offset + 2 * width, ts_dime, width, label="dime", color="orange")
ax.bar(ind - offset + 3 * width, ts_jafa, width, label="jafa", color="indigo")
ax.set_xticks(ind, labels_l)
ax.set_ylabel("log-scaled time (sec)", y=0.45)
ax.set_yscale("log")
fig.legend(
    ncols=4,
    loc="outside lower center",
    frameon=False,
)
fig.savefig(os.path.join(outputs_p, fn), dpi=720)
plt.show()

# %%
