# %%
from __future__ import annotations

import os
from pathlib import Path
import pickle as pkl
from typing import Any, Callable

import mydatasets
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchvision as thv

# %%
os.makedirs("tmp", exist_ok=True)

# %%
_tdata = thv.datasets.Food101(
    "tmp",
    transform=thv.transforms.Compose(
        [
            thv.transforms.PILToTensor(),
            thv.transforms.Grayscale(),
            thv.transforms.Resize((64, 64)),
        ]
    ),
    split="train",
    download=True,
)
_vdata = thv.datasets.Food101(
    "tmp",
    transform=thv.transforms.Compose(
        [
            thv.transforms.PILToTensor(),
            thv.transforms.Grayscale(),
            thv.transforms.Resize((64, 64)),
        ]
    ),
    split="test",
    download=True,
)

# %%
tdata = thd.cat(
    [
        thd.make_tensordict(
            {"xs": _x[:, 0].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _tdata,
            batch_size=36,
            shuffle=False,
            drop_last=False,
            num_workers=36,
        )
    ]
)
_tmpdata = thd.cat(
    [
        thd.make_tensordict(
            {"xs": _x[:, 0].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _vdata,
            batch_size=36,
            shuffle=False,
            drop_last=False,
            num_workers=36,
        )
    ]
)

# %%
_vidxs, _tstidxs = [
    th.as_tensor(_d.indices, dtype=th.long)
    for _d in th_data.random_split(
        th_data.TensorDataset(th.arange(0, len(_tmpdata), dtype=th.long)),
        lengths=(0.5, 0.5),
        generator=th.Generator().manual_seed(279),
    )
]
vdata, tstdata = _tmpdata[_vidxs], _tmpdata[_tstidxs]

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "food101.pkl"
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
