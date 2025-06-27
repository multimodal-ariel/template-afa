# %%
import os
import pickle as pkl

import matplotlib.pyplot as plt
import mydatasets
import numpy as np
import pandas as pd
import sklearn.impute as skl_impute
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th

# %%
tdata_raw: pd.DataFrame = pd.read_csv(
    os.path.join("volvo", "aps_failure_training_set.csv"), na_values="na"
)
_vdata_raw: pd.DataFrame = pd.read_csv(
    os.path.join("volvo", "aps_failure_test_set.csv"), na_values="na"
)
_rg: th.Generator = th.Generator().manual_seed(279)
_shuffle_idxs: np.ndarray = th.randperm(len(_vdata_raw), generator=_rg).numpy()
vdata_raw: pd.DataFrame = _vdata_raw.iloc[_shuffle_idxs[: len(_vdata_raw) // 2]]
tstdata_raw: pd.DataFrame = _vdata_raw.iloc[_shuffle_idxs[len(_vdata_raw) // 2 :]]

# %%
# original dataset without imputation
label_enc = skl_preproc.LabelEncoder()
data_orig = {
    "train": (
        tdata_raw.drop(columns="class").to_numpy(),
        label_enc.fit_transform(tdata_raw["class"]),
    ),
    "valid": (
        vdata_raw.drop(columns="class").to_numpy(),
        label_enc.transform(vdata_raw["class"]),
    ),
    "test": (
        tstdata_raw.drop(columns="class").to_numpy(),
        label_enc.transform(tstdata_raw["class"]),
    ),
}

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "volvo_orig.pkl"
    ),
    mode="wb",
) as f:
    pkl.dump(data_orig, f)

# %%
imputer = skl_impute.KNNImputer(n_neighbors=10)
data = {
    "train": (
        imputer.fit_transform(tdata_raw.drop(columns="class").to_numpy()),
        label_enc.fit_transform(tdata_raw["class"]),
    ),
    "valid": (
        imputer.transform(vdata_raw.drop(columns="class").to_numpy()),
        label_enc.transform(vdata_raw["class"]),
    ),
    "test": (
        imputer.transform(tstdata_raw.drop(columns="class").to_numpy()),
        label_enc.transform(tstdata_raw["class"]),
    ),
}

# %%
with open(
    os.path.join(mydatasets.common.get_datasets_files_root_dir(), "aaco", "volvo.pkl"),
    mode="wb",
) as f:
    pkl.dump(data, f)

# %%
