# %%
import os
import pickle as pkl

import mydatasets.common
import numpy as np
import torchvision.datasets
from skimage.transform import resize

# %%
tdata = torchvision.datasets.MNIST(".", download=True)
vdata = torchvision.datasets.MNIST(".", train=False, download=True)

# %%
# load mnist dataset
x_train = tdata.data.numpy()
y_train = tdata.targets.numpy()
x_test = vdata.data.numpy()
y_test = vdata.targets.numpy()

# %%
xrsz = np.float32(
    np.concatenate([np.reshape(resize(img, (16, 16)), [1, -1]) for img in x_train], 0)
)

xrsz_test = np.float32(
    np.concatenate([np.reshape(resize(img, (16, 16)), [1, -1]) for img in x_test], 0)
)

# %%
Xtrn, Ytrn = (xrsz, y_train)
muX = 0.0  # np.mean(Xtrn, 0, keepdims=True)
stdX = 1.0  # np.std(Xtrn, 0, keepdims=True)
Xtrn = (Xtrn - muX) / stdX
Ntrn, d = Xtrn.shape
Ytrn = np.int32(Ytrn)

# %%
Xval, Yval = (xrsz_test, y_test)
Xval = (Xval - muX) / stdX
Yval = np.int32(Yval)

# %%
Xtst, Ytst = (xrsz_test, y_test)
Xtst = (Xtst - muX) / stdX
Ytst = np.int32(Ytst)

# %%
with open(
    os.path.join(mydatasets.common.get_datasets_files_root_dir(), "aaco", "mnist.pkl"),
    mode="wb",
) as f:
    pkl.dump({"train": (Xtrn, Ytrn), "valid": (Xval, Yval), "test": (Xtst, Ytst)}, f)


# %%
nclass = 10
d = 256
eye_nclass = np.eye(nclass, dtype=np.float32)
Ytrn_onehot = np.concatenate([eye_nclass[np.int64(y), None, :] for y in Ytrn], 0)

# %%
