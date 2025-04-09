# %%
import os

import hydra as hd
import mylib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# %%
metrics_p: str = (
    "experiments/baselines/jafa/outputs/grid/20250327_071151/1/jafa_results.csv"
)

# %%
metrics_df: pd.DataFrame = pd.read_csv(
    os.path.join(mylib.utils.get_project_root_dir(), metrics_p)
)

# %%
metrics_df.rename(columns={"val_acc": "eval/acc"}, inplace=True)

# %%
