# %%
from __future__ import annotations

import os
import pickle as pkl

import mydatasets
import PIL.Image
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchvision.transforms.v2 as thvis_tfv2

# %%
clslbls: list[str] = sorted(os.listdir("traffic"))  # type:ignore

# %%
n: int = sum([len(os.listdir(os.path.join("traffic", _l))) for _l in clslbls])
img_size: int = 128

# %%
xs: th.Tensor = th.empty((n, 3, img_size, img_size), dtype=th.float32)
ys: th.Tensor = th.empty((n,), dtype=th.long)
_i: int = 0
transform_fn = thvis_tfv2.Compose(
    [
        thvis_tfv2.ToImage(),
        thvis_tfv2.Resize((img_size, img_size)),
        thvis_tfv2.ToDtype(th.float32, scale=True),
    ]
)
for _y, _l in enumerate(clslbls):
    _p: str = os.path.join("traffic", _l)
    for _fn in sorted(os.listdir(_p)):
        with PIL.Image.open(os.path.join(_p, _fn)) as _img:
            xs[_i] = transform_fn(_img)
            ys[_i] = _y
        _i = _i + 1

# %%
data = thd.TensorDict({"xs": xs.flatten(1, -1), "ys": ys}).auto_batch_size_(1)

# %%
tidxs, vidxs, tstidxs = [
    _subset.indices
    for _subset in th_data.random_split(
        th.arange(len(data)), (0.8, 0.1, 0.1), generator=th.Generator().manual_seed(279)
    )
]

# %%
tdata = data[tidxs]
vdata = data[vidxs]
tstdata = data[tstidxs]

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "traffic.pkl"
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
