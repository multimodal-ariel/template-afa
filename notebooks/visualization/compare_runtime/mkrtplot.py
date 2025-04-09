# %%
import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

# %%
outputs_p: str = "outputs"
os.makedirs(outputs_p, exist_ok=True)

# %%
# nfeats_aaco = [36.70, 78.900, 6.3, 8.30, 11.30]
# nfeats_tafa = [47.2, 10.89, 6.0, 10.60, 34.79]
fn: str = "time.png"
ts_aaco = [5.9212, 1.2519, 29.280, 29.0495, 26.1010]
ts_tafa = [0.34837, 0.0248, 0.2254, 0.05618, 0.6679]
labels_l = ["big5", "cube", "gas", "grid", "mnist"]

# %%
width = 0.3
ind = np.arange(len(labels_l))

# %%
fig: Figure
ax: Axes
fig, ax = plt.subplots(layout="constrained")
# fig.tight_layout()
fig.set_figheight(2.0)
fig.set_figwidth(0.6 + len(labels_l) * 0.4)
ax.bar(ind, ts_aaco, width, label="aaco")
ax.bar(ind + width, ts_tafa, width, label="ours")
ax.set_xticks(ind + width / 2, labels_l)
ax.set_ylabel("log-scaled time (sec)")
ax.set_yscale("log")
fig.legend(
    ncols=2,
    loc="outside lower center",
    frameon=False,
)
fig.savefig(os.path.join(outputs_p, fn), dpi=720)
plt.show()

# %%
