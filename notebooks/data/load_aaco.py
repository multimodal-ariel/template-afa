# %%
from __future__ import annotations

import mydatasets.aaco

# %%
tcube, vcube, tstcube = mydatasets.aaco.load_aaco_data(
    "cube_20_0.3", to_normalize=False
)

# %%
tgas, vgas, tstgas = mydatasets.aaco.load_aaco_data("gas")

# %%
tgrid, vgrid, tstgrid = mydatasets.aaco.load_aaco_data("grid_data")

# %%
tmnist, vmnist, tstmnist = mydatasets.aaco.load_aaco_data("mnist")

# %%
tbg5, vbg5, tstbg5 = mydatasets.aaco.load_aaco_data("big5_C_cls", to_normalize=True)

# %%
