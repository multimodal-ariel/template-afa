# %%
from __future__ import annotations

import sklearn.feature_selection as skl_fsel
import mydatasets

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("big5_C_cls")

# %%
model = skl_fsel.SelectPercentile().fit(tdata["xs"].numpy(), tdata["ys"].numpy())

# %%
