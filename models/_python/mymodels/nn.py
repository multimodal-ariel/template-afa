from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence, Type, TypeVar

import torch as th
import torchinfo as thinfo

ModuleT = TypeVar("ModuleT", bound=th.nn.Module)


def make_lazy_nnet(
    in_features: int,
    out_features: int,
    layers: Sequence[th.nn.Module] | th.nn.Sequential,
    input_unflatten_shape: Optional[tuple[int, ...]] = None,
) -> th.nn.Sequential:
    nnet: th.nn.Sequential = (
        layers if isinstance(layers, th.nn.Sequential) else th.nn.Sequential(*layers)
    )
    if input_unflatten_shape is not None:
        assert in_features == math.prod(input_unflatten_shape)
        nnet.insert(0, th.nn.Unflatten(dim=1, unflattened_size=input_unflatten_shape))
    nnet.extend([th.nn.Flatten(), th.nn.LazyLinear(out_features=out_features)])
    # sanity check
    lgr = logging.getLogger(make_lazy_nnet.__name__)
    lgr.addHandler(logging.StreamHandler())
    lgr.setLevel(logging.INFO)
    lgr.info(thinfo.summary(nnet, input_size=(in_features,), batch_dim=0))
    return nnet


def make_fcn(
    in_features: int,
    out_features: int,
    layer_specs: Sequence[
        tuple[
            int,
            Optional[Type[th.nn.Module] | Callable[[int], th.nn.Module]],
            Optional[Type[th.nn.Module] | Callable[[], th.nn.Module]],
            Optional[float],
        ]
    ],
) -> th.nn.Sequential:
    """make fully connected network

    Args:
        nnet_in_features (int): input feature size of the FCN
        nnet_out_features (int): output feature size of the FCN
        layer_specs (Sequence[ tuple[ int, Optional[Type[th.nn.Module]  |  Callable[[int], th.nn.Module]], Optional[Type[th.nn.Module]  |  Callable[[], th.nn.Module]], ] ]): a sequence of (hidden_size, make_norm_func, make_activation_func, dropout_p)

    Returns:
        th.nn.Sequential: the fully connected layer
    """
    fcn = th.nn.Sequential()
    for idx, (hsz, make_norm_func, make_activation_func, dropout_p) in enumerate(
        layer_specs
    ):
        idx: int
        hsz: int
        if idx == 0:
            # first layer
            fcn.append(th.nn.Linear(in_features, hsz))
            if make_norm_func is not None:
                fcn.append(make_norm_func(hsz))
        else:
            # all intermediate layer
            fcn.append(th.nn.Linear(layer_specs[idx - 1][0], hsz))
            if make_norm_func is not None:
                fcn.append(make_norm_func(hsz))
        if make_activation_func is not None:
            fcn.append(make_activation_func())
        if dropout_p is not None:
            fcn.append(th.nn.Dropout(p=dropout_p))
        if idx + 1 == len(layer_specs):
            # final output layer
            fcn.append(th.nn.Linear(hsz, out_features))
    return fcn
