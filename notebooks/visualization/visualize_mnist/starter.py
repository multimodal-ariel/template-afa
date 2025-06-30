# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mydatasets
import mylib
import tensordict as thd
import torch as th

# %%
run_p: str = "experiments/make_template/outputs/mnist_cnnet/20250331_133521/0"
run_p = os.path.join(mylib.utils.get_project_root_dir(), run_p)

# %%
# keys:
#   - "xs": (n, n_covs) for mnist, n_covs = 16*16 = 256
#   - "ys": (n, )
_tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("mnist", to_normalize=True)
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(run_p, "tdata_shuffle_idxs.pt"), weights_only=False
)
tdata: thd.TensorDict = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]

# %%
# (n_tmpls, n_covs)
tmpls: th.Tensor = th.load(os.path.join(run_p, "tmpls.pt"), weights_only=False)
# (len(tdata), )
# keys:
#   - "cels": (len(tdata), n_tmpls)
#   - "pyhats": (len(tdata), n_tmpls, n_labels)
#   - "rwds": (len(tdata), n_tmpls)
tpcomp: thd.TensorDict = th.load(os.path.join(run_p, "tpcomp.pt"), weights_only=False)

# %%
plt.imshow(th.reshape(tdata["xs"][th.randint(len(tdata), ())], (16, 16)))
plt.show()
plt.close()

# %%
plt.imshow(th.reshape(tmpls[th.randint(len(tmpls), ())], (16, 16)))
plt.show()
plt.close()

# %%
