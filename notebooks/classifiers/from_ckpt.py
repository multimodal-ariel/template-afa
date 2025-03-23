# %%
from __future__ import annotations
import os

import _tmplfns
import lightning as pl
import mydatasets.aaco
import mymodels.classifiers
import mymodels.nn
import pandas as pd
import torch as th
from omegaconf import OmegaConf
import hydra as hd

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = "../../"
run_p: str = (
    "experiments/pretrain/nnet_subset_feature_classifier/outputs/mnist/20250313_000931/"
)
# run_p: str = (
#     "experiments/pretrain/nnet_subset_feature_classifier/outputs/big5/20250312_224514"
# )

# %%
run_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, run_p, ".hydra", "config.yaml"))

# %%
tdata, vdata, tstdata = hd.utils.instantiate(run_cfg.data)
n_covs: int = tdata["xs"].shape[1]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
tmpls: th.Tensor = _tmplfns.make_feature_masks(
    n_covs=n_covs, n_masks=run_cfg.n_masks, min_features=1, max_features=None
)

# %%
plf = pl.Fabric(accelerator="auto")

# %%
classifier = (
    mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
        nnet=hd.utils.instantiate(
            run_cfg.nnet,
            in_features=n_covs * 2,
            out_features=n_labels,
        ),
        xs_train=tdata["xs"].numpy(),
        ys_train=tdata["ys"].numpy(),
        fit_kwargs=hd.utils.instantiate(run_cfg.nnet_fit_cfg),
        state_dict_p=os.path.join(PROJ_ROOT, run_p, "classifier.pt"),
    ).to(device=plf.device)
)

# %%
metrics_d: dict[str, float] = classifier.fit_(tmpls)
print(pd.Series(metrics_d))

# %%
metrics_d: dict[str, float] = classifier.evaluate(vdata, tmpls, None)
print(pd.Series(metrics_d))

# %%
# tmpls: th.Tensor = _tmplfns.make_template_candidates(
#     n_covs=n_covs, init_fidx=35, n_cands_targ=10_000, min_features=1, max_features=None
# )

# # %%
# # metrics_d: dict[str, float] = classifier.fit_(tmpls)
# # print(pd.Series(metrics_d))

# # %%
# metrics_d: dict[str, float] = classifier.evaluate(vdata, tmpls, None)
# print(pd.Series(metrics_d))

# # %%
