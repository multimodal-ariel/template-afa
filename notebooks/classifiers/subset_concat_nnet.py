# %%
from __future__ import annotations

import os

import lightning as pl
import mydatasets
import mymodels
import pandas as pd
import tafalib
import torch as th

# %%
PROJ_ROOT: str = "../../"

# %%
# tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("mnist")
# n_covs: int = tdata["xs"].shape[1]
# n_labels: int = len(th.unique(tdata["ys"]))

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("traffic", to_normalize=False)
n_covs: int = tdata["xs"].shape[1]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
tmpls: th.Tensor = tafalib.makers.candidates.make_feature_masks(
    n_covs=n_covs, n_masks=1_000, min_features=1, max_features=None, generator=None
)

# %%
plf = pl.Fabric(accelerator="cpu")

# %%
classifier = mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier(
    nnet=mymodels.nn.make_fcn(
        in_features=n_covs * 2,
        out_features=n_labels,
        layer_specs=[(512, None, th.nn.ReLU, None), (512, None, th.nn.ReLU, None)],
    ),
    xs_train=tdata["xs"].numpy(),
    ys_train=tdata["ys"].numpy(),
    fit_kwargs=mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier._FitKwargs(
        opt_type=th.optim.Adam,
        opt_kwargs=dict(),
        n_iter=1,
        bsz=1024,
    ),
).to(device=plf.device)

# %%
metrics_d: dict[str, float] = classifier.fit_(tmpls)
print(pd.Series(metrics_d))

# %%
metrics_d: dict[str, float] = classifier.evaluate(vdata, tmpls, None)
print(pd.Series(metrics_d))

# %%
tmpls: th.Tensor = tafalib.makers.candidates.make_template_candidates(
    n_covs=n_covs,
    init_fidx=35,
    n_cands_targ=10_000,
    min_features=1,
    max_features=None,
    generator=None,
)

# %%
metrics_d: dict[str, float] = classifier.fit_(tmpls)
print(pd.Series(metrics_d))

# %%
metrics_d: dict[str, float] = classifier.evaluate(vdata, tmpls, None)
print(pd.Series(metrics_d))

# %%
