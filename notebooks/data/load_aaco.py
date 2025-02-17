# %%
from __future__ import annotations

import mydatasets.aaco

# %%
tcube, vcube, tstcube = mydatasets.aaco.load_aaco_data(
    "cube_20_0.3", to_normalize=False
)
print(tcube["xs"].shape)

# %%
tgas, vgas, tstgas = mydatasets.aaco.load_aaco_data("gas")
print(tgas["xs"].shape)

# %%
tgrid, vgrid, tstgrid = mydatasets.aaco.load_aaco_data("grid_data")
print(tgrid["xs"].shape)

# %%
tmnist, vmnist, tstmnist = mydatasets.aaco.load_aaco_data("mnist")
print(tmnist["xs"].shape)

# %%
tbg5, vbg5, tstbg5 = mydatasets.aaco.load_aaco_data("big5_C_cls", to_normalize=True)
print(tbg5["xs"].shape)

# %%
