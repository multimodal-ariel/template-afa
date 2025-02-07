from __future__ import annotations

import os

import mydatasets.common
import numpy as np
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th


def load_aaco_data(
    name: str, to_normalize: bool = True
) -> tuple[thd.TensorDict, thd.TensorDict]:
    data: dict[str, tuple[np.ndarray, np.ndarray]] = np.load(
        os.path.join(
            mydatasets.common.get_datasets_files_root_dir(), "aaco", f"{name}.pkl"
        ),
        allow_pickle=True,
    )
    xst: np.ndarray = data["train"][0]
    yst: np.ndarray = data["train"][1].flatten()
    xsv: np.ndarray = data["valid"][0]
    ysv: np.ndarray = data["valid"][1].flatten()
    if to_normalize:
        nmlr = skl_preproc.StandardScaler()
        xst = nmlr.fit_transform(xst)
        xsv = nmlr.transform(xsv)
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
    return tdata, vdata
