# %%
from __future__ import annotations

import imblearn as imbl
import mydatasets.aaco

# %%
tbg5, vbg5, tstbg5 = mydatasets.aaco.load_aaco_data("big5_C_cls", to_normalize=True)
txs_n, tys_n = tbg5["xs"].numpy(), tbg5["ys"].numpy()

# %%
cnn_rsmp = imbl.under_sampling.CondensedNearestNeighbour(n_jobs=-1)
cnn_rsmp.fit_resample(txs_n, tys_n)

# %%
len(cnn_rsmp.sample_indices_)

# %%
tgas, vgas, tstgas = mydatasets.aaco.load_aaco_data("gas", to_normalize=True)
txs_n, tys_n = tgas["xs"].numpy(), tgas["ys"].numpy()

# %%
cnn_rsmp = imbl.under_sampling.CondensedNearestNeighbour(n_jobs=-1)
cnn_rsmp.fit_resample(txs_n, tys_n)

# %%
len(cnn_rsmp.sample_indices_)

# %%
