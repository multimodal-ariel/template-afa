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
print(tmnist["xs"].shape, tmnist["ys"].shape)

# %%
tfmnist, vfmnist, tstfmnist = mydatasets.aaco.load_aaco_data("fashion-mnist")
print(tfmnist["xs"].shape, tfmnist["ys"].shape)

# %%
tgtsrb, vgtsrb, tstgtsrb = mydatasets.aaco.load_aaco_data("gtsrb-gray")
print(tgtsrb["xs"].shape, tgtsrb["ys"].shape)

# %%
tfood, vfood, tstfood = mydatasets.aaco.load_aaco_data("food101")
print(tfood["xs"].shape, tfood["ys"].shape)

# %%
tbg5, vbg5, tstbg5 = mydatasets.aaco.load_aaco_data("big5_C_cls", to_normalize=True)
print(tbg5["xs"].shape)

# %%
tcharfont, vcharfont, tstcharfont = mydatasets.aaco.load_aaco_data(
    "charfont-1500", to_normalize=True
)
print(tcharfont["xs"].shape)

# %%
tvolvo, vvolvo, tstvolvo = mydatasets.aaco.load_aaco_data("volvo", to_normalize=True)
print(tvolvo["xs"].shape)

# %%
tctslice, vctslice, tstctslice = mydatasets.aaco.load_aaco_data("ct-slice-rel-loc")
print(tctslice["xs"].shape)

# %%
tctengine, vctengine, tstctengine = mydatasets.aaco.load_aaco_data("engine-fault")
print(tctengine["xs"].shape)

# %%
