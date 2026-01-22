# %%
from __future__ import annotations

import os
from typing import Sequence

import _classifiers
import mydatasets
import numpy as np
import pandas as pd
import sklearn.tree as skl_tree
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm

# %%
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(
    "engine-fault", to_normalize=True
)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
n_masks: int = 40

# %%
fms: th.Tensor = tafalib.makers.candidates.make_feature_masks(
    n_covs=n_covs, n_masks=n_masks, min_features=1, max_features=None, generator=None
)

# %%
# split training data into two for classifier and afa
_tdata_shuffle_idxs = th.randperm(
    len(_tdata), generator=th.Generator().manual_seed(279)
)
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
clf = _classifiers.EngineFaultDecisionTreeClassifier(
    dtc_kwargs={
        "max_depth": 6,
        "splitter": "best",
        "criterion": "log_loss",
        "random_state": 279,
        "ccp_alpha": 0.025,
    },
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
)

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
_txs: th.Tensor = tdata["xs"][:, None, :].expand(-1, n_masks, -1).flatten(0, 1)
_tfms: th.Tensor = fms[None, :, :].expand(len(tdata), -1, -1).flatten(0, 1)
_tys: th.Tensor = tdata["ys"][:, None].expand(-1, n_masks).flatten(0, 1)
tpyhats: th.Tensor = th.as_tensor(clf.predict_proba(_txs, _tfms))
metrics_func.reset()
metrics_func.update(tpyhats.to(device="cpu"), _tys)
tmetrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
print(pd.Series(tmetrics_d))

# %%
_vxs: th.Tensor = vdata["xs"][:, None, :].expand(-1, n_masks, -1).flatten(0, 1)
_vfms: th.Tensor = fms[None, :, :].expand(len(vdata), -1, -1).flatten(0, 1)
_vys: th.Tensor = vdata["ys"][:, None].expand(-1, n_masks).flatten(0, 1)
vpyhats: th.Tensor = th.as_tensor(clf.predict_proba(_vxs, _vfms))
metrics_func.reset()
metrics_func.update(vpyhats.to(device="cpu"), _vys)
vmetrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
print(pd.Series(vmetrics_d))

# %%
_tstxs: th.Tensor = tstdata["xs"][:, None, :].expand(-1, n_masks, -1).flatten(0, 1)
_tstfms: th.Tensor = fms[None, :, :].expand(len(tstdata), -1, -1).flatten(0, 1)
_tstys: th.Tensor = tstdata["ys"][:, None].expand(-1, n_masks).flatten(0, 1)
tstpyhats: th.Tensor = th.as_tensor(clf.predict_proba(_tstxs, _tstfms))
metrics_func.reset()
metrics_func.update(tstpyhats.to(device="cpu"), _tstys)
tstmetrics_d: dict[str, float] = {
    k: v.item() for k, v in metrics_func.compute().items()
}
print(pd.Series(tstmetrics_d))


# %%
def plot_tree(
    clf: _classifiers.EngineFaultDecisionTreeClassifier,
    subset_features: Sequence[str],
    save_p: str,
):
    _key: tuple[int, ...] = tuple(
        np.sum(
            np.stack([clf.FEATURE_NAMES == _n for _n in subset_features], axis=0),
            axis=0,
            keepdims=False,
        ).tolist()
    )
    _clf = clf[_key]
    skl_tree.export_graphviz(
        _clf,
        out_file=f"{save_p}.dot",
        feature_names=_clf.feature_names_in_,
        class_names=[
            "normal",
            "rich mixture",
            "lean mixture",
            "low voltage",
        ],
        rotate=True,
        impurity=False,
        proportion=True,
        label="root",
    )


# %%
out_p: str = "outputs/engine-dtc"
os.makedirs(out_p, exist_ok=True)

# %%
# feature_names = [
#     "MAP",
#     "TPS",
#     "Force",
#     "Power",
#     "RPM",
#     "Consumption L/H",
#     "Consumption L/100KM",
#     "Speed",
#     "CO",
#     "HC",
#     "CO2",
#     "O2",
#     "Lambda",
#     "AFR",
# ]

# %%
plot_tree(
    clf,
    ["Consumption L/100KM"],
    os.path.join(out_p, "lper100km"),
)

# %%
# beta0.24, accuracy 12
_out_p = os.path.join(out_p, "beta0.24-acc-12")
os.makedirs(_out_p, exist_ok=True)
plot_tree(
    clf,
    ["Consumption L/100KM", "CO"],
    os.path.join(_out_p, "lper100km-co"),
)
plot_tree(
    clf,
    ["Consumption L/100KM", "Speed", "O2"],
    os.path.join(_out_p, "lper100km-kph-o2"),
)
plot_tree(
    clf,
    ["Consumption L/100KM", "HC"],
    os.path.join(_out_p, "lper100km-hc"),
)
plot_tree(
    clf,
    ["Consumption L/100KM", "RPM", "CO", "Speed", "O2"],
    os.path.join(_out_p, "rpm-lper100km-speed-co-o2"),
)
plot_tree(
    clf,
    ["MAP", "Consumption L/100KM", "CO"],
    os.path.join(_out_p, "map-lper100km-co"),
)
plot_tree(
    clf,
    [
        "Force",
        "RPM",
        "Consumption L/100KM",
        "Speed",
        "CO",
        "O2",
    ],
    os.path.join(_out_p, "force-rpm-lper100km-speed-co-o2"),
)

# %%
