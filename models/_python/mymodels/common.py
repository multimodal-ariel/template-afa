from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch as th
import torch.utils.dlpack

if TYPE_CHECKING:
    import cupy as cp


def to_cp_or_np(inputs: th.Tensor) -> np.ndarray | cp.ndarray:
    try:
        import cupy as cp
    except ImportError:
        return inputs.numpy(force=True)
    inputs_n = (
        cp.from_dlpack(torch.utils.dlpack.to_dlpack(inputs))  # type: ignore
        if inputs.device.type == "cuda"
        else inputs.numpy(force=True)
    )
    return inputs_n
