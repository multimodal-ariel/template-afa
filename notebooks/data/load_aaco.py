# %%
from __future__ import annotations

import mydatasets.aaco

# %%
tcube, vcube = mydatasets.aaco.load_aaco_data("cube_20_0.3", to_normalize=False)

# %%
tgas, vgas = mydatasets.aaco.load_aaco_data("gas")

# %%
tgrid, vgrid = mydatasets.aaco.load_aaco_data("grid_data")

# %%
tmnist, vmnist = mydatasets.aaco.load_aaco_data("mnist")

# %%
