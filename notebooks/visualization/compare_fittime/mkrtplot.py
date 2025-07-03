# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# %%
outputs_p: str = "outputs"
os.makedirs(outputs_p, exist_ok=True)

# %%
# nfeats_aaco = [36.70, 78.900, 6.3, 8.30, 11.30]
# nfeats_tafa = [47.2, 10.89, 6.0, 10.60, 34.79]
fn: str = "fit_time.png"
ts_aaco = [0, 0, 0, 0, 0]
ts_tafa = [
    pd.Timestamp(year=2025, month=3, day=18, hour=14, minute=53, second=29)
    - pd.Timestamp(year=2025, month=3, day=18, hour=14, minute=41, second=21),
    pd.Timestamp(year=2025, month=3, day=18, hour=23, minute=17, second=24)
    - pd.Timestamp(year=2025, month=3, day=18, hour=22, minute=54, second=16),
    pd.Timestamp(year=2025, month=3, day=24, hour=23, minute=21, second=52)
    - pd.Timestamp(year=2025, month=3, day=24, hour=22, minute=47, second=43),
    pd.Timestamp(year=2025, month=3, day=25, hour=21, minute=45, second=21)
    - pd.Timestamp(year=2025, month=3, day=25, hour=21, minute=36, second=28),
    pd.Timestamp(year=2025, month=3, day=26, hour=9, minute=50, second=18)
    - pd.Timestamp(year=2025, month=3, day=26, hour=00, minute=38, second=25),
]
ts_dime = [
    pd.Timedelta(hours=1.237),
    pd.Timedelta(minutes=13.25),
    pd.Timedelta(hours=2.019),
    pd.Timedelta(minutes=10.61),
    pd.Timedelta(hours=6.317),
]
ts_jafa = [
    pd.Timestamp(year=2025, month=3, day=30, hour=21, minute=4, second=11)
    - pd.Timestamp(year=2025, month=3, day=30, hour=18, minute=46, second=11),
    pd.Timestamp(year=2025, month=3, day=30, hour=23, minute=15, second=14)
    - pd.Timestamp(year=2025, month=3, day=30, hour=21, minute=11, second=51),
    pd.Timestamp(year=2025, month=3, day=31, hour=4, minute=17, second=31)
    - pd.Timestamp(year=2025, month=3, day=31, hour=00, minute=47, second=7),
    pd.Timestamp(year=2025, month=3, day=27, hour=12, minute=30, second=54)
    - pd.Timestamp(year=2025, month=3, day=27, hour=7, minute=11, second=55),
    pd.Timestamp(year=2025, month=4, day=14, hour=12, minute=57, second=37)
    - pd.Timestamp(year=2025, month=4, day=8, hour=12, minute=30, second=11),
]
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
ax.bar(
    ind - offset + 1 * width,
    [_tdelta / pd.Timedelta(hours=1) for _tdelta in ts_tafa],
    width,
    label="ours",
    color="blue",
)
ax.bar(
    ind - offset + 2 * width,
    [_tdelta / pd.Timedelta(hours=1) for _tdelta in ts_dime],
    width,
    label="dime",
    color="orange",
)
ax.bar(
    ind - offset + 3 * width,
    [_tdelta / pd.Timedelta(hours=1) for _tdelta in ts_jafa],
    width,
    label="jafa",
    color="indigo",
)
ax.set_xticks(ind, labels_l)
ax.set_ylabel("log-scaled time (hour)", y=0.45)
ax.set_yscale("log")
fig.legend(
    ncols=4,
    loc="outside lower center",
    frameon=False,
)
fig.savefig(os.path.join(outputs_p, fn), dpi=720)
plt.show()

# %%
