from __future__ import annotations

import os

import mydatasets.common
import numpy as np
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th


def load_aaco_data(
    name: str, to_normalize: bool = True
) -> tuple[thd.TensorDict, thd.TensorDict, thd.TensorDict]:
    data: dict[str, tuple[np.ndarray, np.ndarray]] = np.load(
        os.path.join(
            mydatasets.common.get_datasets_files_root_dir(), "aaco", f"{name}.pkl"
        ),
        allow_pickle=True,
    )
    xst: np.ndarray = data["train"][0].astype(np.float32)
    yst: np.ndarray = data["train"][1].flatten()
    xsv: np.ndarray = data["valid"][0].astype(np.float32)
    ysv: np.ndarray = data["valid"][1].flatten()
    xstst: np.ndarray = data["test"][0].astype(np.float32)
    ystst: np.ndarray = data["test"][1].flatten()
    if to_normalize:
        nmlr = skl_preproc.StandardScaler(copy=False)
        xst = nmlr.fit_transform(xst)
        xsv = nmlr.transform(xsv)
        xstst = nmlr.transform(xstst)
    tdata = thd.TensorDict(
        {
            "xs": th.as_tensor(xst, dtype=th.float32),
            "ys": th.as_tensor(yst, dtype=th.long),
        }
    ).auto_batch_size_(1)
    vdata = thd.TensorDict(
        {
            "xs": th.as_tensor(xsv, dtype=th.float32),
            "ys": th.as_tensor(ysv, dtype=th.long),
        }
    ).auto_batch_size_(1)
    tstdata = thd.TensorDict(
        {
            "xs": th.as_tensor(xstst, dtype=th.float32),
            "ys": th.as_tensor(ystst, dtype=th.long),
        }
    ).auto_batch_size_(1)
    return tdata, vdata, tstdata
