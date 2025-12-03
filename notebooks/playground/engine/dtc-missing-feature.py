# %%
from __future__ import annotations

import mydatasets
import pandas as pd
import sklearn.tree as skl_tree
import torch as th
import torchmetrics as thm

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(
    "engine-fault", to_normalize=False
)
n_covs: int = tdata["xs"].shape[1]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
n_masks: int = 500
_txs: th.Tensor = tdata["xs"][:, None, :].expand(-1, n_masks, -1)
_tfms: th.Tensor = th.randint(0, 2, (len(tdata), n_masks, n_covs)).to(dtype=th.bool)
txs: th.Tensor = th.cat(
    (th.where(_tfms.to(dtype=th.bool), _txs, th.nan).flatten(0, 1), tdata["xs"]), dim=0
)
tys: th.Tensor = th.cat(
    (tdata["ys"][:, None].expand(-1, 500).flatten(0, 1), tdata["ys"]), dim=0
)

# %%
clf = skl_tree.DecisionTreeClassifier(
    max_depth=13, splitter="best", criterion="log_loss"
)
clf.fit(txs.numpy(), tys.numpy())

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
tpyhats_rndms: th.Tensor = th.as_tensor(
    clf.predict_proba(
        th.where(
            th.randint_like(tdata["xs"], 0, 2).to(dtype=th.bool), tdata["xs"], th.nan
        ).numpy()
    )
)
metrics_func.reset()
metrics_func.update(tpyhats_rndms.to(device="cpu"), tdata["ys"].to(device="cpu"))
trndmsmetrics_d: dict[str, float] = {
    k: v.item() for k, v in metrics_func.compute().items()
}
print(pd.Series(trndmsmetrics_d))

# %%
vpyhats_rndms: th.Tensor = th.as_tensor(
    clf.predict_proba(
        th.where(
            th.randint_like(vdata["xs"], 0, 2).to(dtype=th.bool), vdata["xs"], th.nan
        ).numpy()
    )
)
metrics_func.reset()
metrics_func.update(vpyhats_rndms.to(device="cpu"), vdata["ys"].to(device="cpu"))
vrndmsmetrics_d: dict[str, float] = {
    k: v.item() for k, v in metrics_func.compute().items()
}
print(pd.Series(vrndmsmetrics_d))

# %%
tstpyhats_rndms: th.Tensor = th.as_tensor(
    clf.predict_proba(
        th.where(
            th.randint_like(tstdata["xs"], 0, 2).to(dtype=th.bool),
            tstdata["xs"],
            th.nan,
        ).numpy()
    )
)
metrics_func.reset()
metrics_func.update(tstpyhats_rndms.to(device="cpu"), tstdata["ys"].to(device="cpu"))
tstrndmsmetrics_d: dict[str, float] = {
    k: v.item() for k, v in metrics_func.compute().items()
}
print(pd.Series(tstrndmsmetrics_d))

# %%
skl_tree.plot_tree(
    clf,
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

# %%
