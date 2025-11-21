# %%
from __future__ import annotations

import os
import pickle as pkl

import mydatasets
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchvision as thv

# %%
_tdata = thv.datasets.FashionMNIST(".", download=True)
_vdata = thv.datasets.FashionMNIST(".", train=False, download=True)

# %%
_tmpdata: thd.TensorDict = thd.make_tensordict(
    {
        "xs": _tdata.data.flatten(1, 2).to(dtype=th.float32),
        "ys": _tdata.targets.flatten(),
    },
    batch_size=(len(_tdata),),
)
vdata: thd.TensorDict = thd.make_tensordict(
    {
        "xs": _vdata.data.flatten(1, 2).to(dtype=th.float32),
        "ys": _vdata.targets.flatten(),
    },
    batch_size=(len(_vdata),),
)

# %%
_tidxs, _tstidxs = [
    th.as_tensor(_d.indices, dtype=th.long)
    for _d in th_data.random_split(
        th_data.TensorDataset(th.arange(0, len(_tmpdata), dtype=th.long)),
        lengths=(50_000, 10_000),
        generator=th.Generator().manual_seed(279),
    )
]
tdata, tstdata = _tmpdata[_tidxs], _tmpdata[_tstidxs]

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "fashion-mnist.pkl"
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

# %%
