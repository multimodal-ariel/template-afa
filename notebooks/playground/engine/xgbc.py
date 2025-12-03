# %%
from __future__ import annotations

import os
import pickle as pkl

import graphviz
import matplotlib.pyplot as plt
import mydatasets
import numpy as np
import pandas as pd
import sklearn.compose as skl_compose
import sklearn.preprocessing as skl_preproc
import sklearn.tree as skl_tree
import tabulate as tbl
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import xgboost as xgbst

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(
    "engine-fault", to_normalize=False
)

# %%
sclr = skl_preproc.MinMaxScaler()
tdata["xs"] = th.as_tensor(sclr.fit_transform(tdata["xs"].numpy()))
vdata["xs"] = th.as_tensor(sclr.transform(vdata["xs"].numpy()))
tstdata["xs"] = th.as_tensor(sclr.transform(tstdata["xs"].numpy()))

# %%
clf = xgbst.XGBClassifier(max_depth=6)
clf.fit(
    tdata["xs"].numpy(),
    tdata["ys"].numpy(),
    eval_set=(
        (tdata["xs"].numpy(), tdata["ys"].numpy()),
        (vdata["xs"].numpy(), vdata["ys"].numpy()),
    ),
)

# %%
n_labels: int = len(th.unique(tdata["ys"]))

# %%
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)

# %%
tpyhats: th.Tensor = th.as_tensor(clf.predict_proba(tdata["xs"].numpy()))
metrics_func.reset()
metrics_func.update(tpyhats.to(device="cpu"), tdata["ys"].to(device="cpu"))
tmetrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
print(pd.Series(tmetrics_d))

# %%
vpyhats: th.Tensor = th.as_tensor(clf.predict_proba(vdata["xs"].numpy()))
metrics_func.reset()
metrics_func.update(vpyhats.to(device="cpu"), vdata["ys"].to(device="cpu"))
vmetrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
print(pd.Series(vmetrics_d))

# %%
tstpyhats: th.Tensor = th.as_tensor(clf.predict_proba(tstdata["xs"].numpy()))
metrics_func.reset()
metrics_func.update(tstpyhats.to(device="cpu"), tstdata["ys"].to(device="cpu"))
tstmetrics_d: dict[str, float] = {
    k: v.item() for k, v in metrics_func.compute().items()
}
print(pd.Series(tstmetrics_d))

# %%
xgbst.plot_tree(
    clf,
    num_trees=0,
    rankdir="LR",
    feature_names=[
        "MAP",
        "TPS",
        "Force",
        "Power",
        "RPM",
        "Consumption L/H",
        "Consumption L/100KM",
        "Speed",
        "CO",
        "HC",
        "CO2",
        "O2",
        "Lambda",
        "AFR",
    ],
    class_names=[
        "normal",
        "rich mixture",
        "lean mixture",
        "low voltage",
    ],
)
plt.show()

# %%
