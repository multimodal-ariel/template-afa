# %%
from __future__ import annotations

import mydatasets
import pandas as pd
import sklearn.tree as skl_tree
import tensordict as thd
import torch as th
import torchmetrics as thm
import _classifiers
import tafalib

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
clf = _classifiers.EngineFaultXGBClassifier(
    xgb_kwargs={
        "n_estimators": 8,
        "max_depth": 8,
    },
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    fraction_training_data_per_split=1.0,
    n_splits=16,
    n_tmpl_per_instance=250,
)
clf.fit_(None)

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
