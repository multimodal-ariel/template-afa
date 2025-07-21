"""Metrics dict so that we can use the model names as string in arguments
to easily select a model from the command line.
"""

from __future__ import annotations

from typing import Type

from .base import BaseModel
from .dime import DIME
from .eddi import EDDI
from .fixed_mlp import FixedMLP
from .gdfs import GDFS
from .opportunistic_rl import OpportunisticRL
from .our_model import OurModel
from .vae import VAE

models_dict: dict[str, Type[BaseModel]] = {
    "dime": DIME,
    "eddi": EDDI,
    "fixed_mlp": FixedMLP,
    "gdfs": GDFS,
    "opportunistic": OpportunisticRL,
    "ours": OurModel,
    "vae": VAE,
}
