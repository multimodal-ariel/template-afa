# %%
import mydatasets.aaco
import numpy as np
import torch as th
import xgboost as xgbst

# %%
th.autograd.set_grad_enabled(False)

# %%notebooks/baselines/aco/train_arbitrary_subset_classifier_big5.py
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("fashion-mnist-16x16")

# %%
# ### Read in Cube data
# cube = np.load("data/cube_20_0.3.pkl", allow_pickle=True)

# cube_train = cube.get("train")
# cube_train_X = th.from_numpy(cube_train[0])
# cube_train_y = th.nn.functional.one_hot(th.from_numpy(cube_train[1]).long())

# cube_val = cube.get("valid")
# cube_val_X = th.from_numpy(cube_val[0])
# cube_val_y = th.nn.functional.one_hot(th.from_numpy(cube_val[1]).long())

# %%
### Read in data
txs: th.Tensor = tdata["xs"]
tys: th.Tensor = th.nn.functional.one_hot(tdata["ys"])

vxs: th.Tensor = vdata["xs"]
vys: th.Tensor = th.nn.functional.one_hot(vdata["ys"])

# %%
n_covs: int = txs.shape[1]

# %%
# Generate 10,000,000 random masks
np.random.seed(29)

Rall = 10

Ball = np.concatenate(
    [
        np.sum(
            np.random.permutation(np.eye(n_covs))[:, : np.random.randint(n_covs)],
            1,
            keepdims=True,
        )
        for _ in range(Rall)
    ],
    1,
)
Ball_t = th.as_tensor(Ball, dtype=th.float32)

# %%
# Set each 7th variable (index 6) each to 1 (only consider masks that select feature 7
# Ball[:, 6] = 1.0

# %%
# Train the classifier in these 10,000,000 random masks
data_x_train = (
    th.cat(
        [
            txs[:, None, :] * Ball_t.T[None, :, :] - (1 - Ball_t.T[None, :, :]) * 10,
            Ball_t.T[None, :, :].expand(len(txs), -1, -1),
        ],
        dim=2,
    )
    .flatten(0, 1)
    .numpy()
)
data_y_train = tys[:, None, :].expand(-1, Rall, -1).flatten(0, 1).argmax(dim=1).numpy()
# data_x_train = np.hstack(
#     (txs.repeat(1000, 1).detach().numpy() * Ball.T - (1 - Ball.T) * 10, Ball.T)
# )
# data_y_train = tys.repeat(1000, 1).argmax(dim=1).detach().numpy()

# %%
model_xgb_arb = xgbst.XGBClassifier(
    n_estimators=50, max_depth=20, random_state=29, n_jobs=-1
)
model_xgb_arb.fit(data_x_train, data_y_train)

# %%
# Save the model
model_xgb_arb.save_model("fashion-16x16_xgb_classifier_arb_subsets.json")

# %%
# Print
print("finished")

# %%
