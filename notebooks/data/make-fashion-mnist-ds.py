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
_tdata = thv.datasets.FashionMNIST(
    ".",
    transform=thv.transforms.Compose(
        [
            thv.transforms.PILToTensor(),
            thv.transforms.Grayscale(),
            thv.transforms.Resize((16, 16)),
        ]
    ),
    download=True,
)
_vdata = thv.datasets.FashionMNIST(
    ".",
    transform=thv.transforms.Compose(
        [
            thv.transforms.PILToTensor(),
            thv.transforms.Grayscale(),
            thv.transforms.Resize((16, 16)),
        ]
    ),
    train=False,
    download=True,
)

# %%
tdata = thd.cat(
    [
        thd.make_tensordict(
            {"xs": _x[:, 0, :, :].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _tdata,
            batch_size=36,
            shuffle=False,
            drop_last=False,
            num_workers=36,
        )
    ],
    dim=0,
)
_tmpdata = thd.cat(
    [
        thd.make_tensordict(
            {"xs": _x[:, 0, :, :].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _vdata,
            batch_size=36,
            shuffle=False,
            drop_last=False,
            num_workers=8,
        )
    ],
    dim=0,
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
        mydatasets.common.get_datasets_files_root_dir(),
        "aaco",
        "fashion-mnist-16x16.pkl",
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
