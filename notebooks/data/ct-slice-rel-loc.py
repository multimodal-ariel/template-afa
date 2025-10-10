# %%
from __future__ import annotations

import os
import pickle as pkl

import mydatasets
import mylib
import pandas as pd
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th
import torch.utils.data as th_data

# %%
rg = th.Generator().manual_seed(279)

# %%
# https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis
data_df: pd.DataFrame = pd.read_csv(
    os.path.join(
        mylib.utils.get_project_root_dir(),
        "notebooks/data/ct-slice-rel-loc/slice_localization_data.csv",
    )
)
datadf = data_df.iloc[th.randperm(len(data_df), generator=rg).numpy()]

# %%
xs: th.Tensor = th.as_tensor(
    skl_preproc.StandardScaler().fit_transform(
        data_df.drop(columns=["reference", "patientId"]).to_numpy()
    ),
    dtype=th.float32,
)
targets: th.Tensor = th.as_tensor(
    data_df["reference"].to_numpy(), dtype=th.float32
).flatten()

# %%
labels: th.Tensor = th.empty_like(targets, dtype=th.long).fill_(3)
for _i, _thrd in enumerate(
    th.quantile(targets, th.tensor([0.75, 0.5, 0.25])).flatten()
):
    labels[targets < _thrd] = th.min(labels).item() - 1

# %%
data: thd.TensorDict = thd.make_tensordict(
    {"xs": xs, "ys": labels}, batch_size=(len(data_df),)
)

# %%
tidxs, vidxs, tstidxs = [
    _subset.indices
    for _subset in th_data.random_split(
        th.arange(len(data_df)),
        (0.8, 0.1, 0.1),
        generator=rg,
    )
]

# %%
tdata = data[tidxs]
vdata = data[vidxs]
tstdata = data[tstidxs]

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "ct-slice-rel-loc.pkl"
    ),
    mode="wb",
) as f:
    pkl.dump(
        {
            "train": (tdata["xs"].numpy(), tdata["ys"].numpy()),
            "valid": (vdata["xs"].numpy(), vdata["ys"].numpy()),
            "test": (tstdata["xs"].numpy(), tstdata["ys"].numpy()),
        },
        f,
    )

# %%
