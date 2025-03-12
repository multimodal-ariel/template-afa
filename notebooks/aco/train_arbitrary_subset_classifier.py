# %%
import mydatasets.aaco
import numpy as np
import torch as th
import xgboost as xgbst

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("cube_20_0.3")

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
### Read in Cube data
cube_train_X = tdata["xs"]
cube_train_y = th.nn.functional.one_hot(tdata["ys"])

cube_val_X = vdata["xs"]
cube_val_y = th.nn.functional.one_hot(vdata["ys"])

# %%
# Generate 10,000,000 random masks
np.random.seed(29)

Rall = 10000000

Ball = np.concatenate(
    [
        np.sum(
            np.random.permutation(np.eye(20))[:, : np.random.randint(20)],
            1,
            keepdims=True,
        )
        for _ in range(Rall)
    ],
    1,
)

# %%
# Set each 7th variable (index 6) each to 1 (only consider masks that select feature 7
Ball[:, 6] = 1.0

# %%
# Train the classifier in these 10,000,000 random masks
data_x_train = np.hstack(
    (cube_train_X.repeat(1000, 1).detach().numpy() * Ball.T - (1 - Ball.T) * 10, Ball.T)
)
data_y_train = cube_train_y.repeat(1000, 1).argmax(dim=1).detach().numpy()


# %%
model_xgb_arb = xgbst.XGBClassifier(
    n_estimators=200, max_depth=20, random_state=29, n_jobs=-1
)
model_xgb_arb.fit(data_x_train, data_y_train)

# %%
# Save the model
model_xgb_arb.save_model("cube_xgb_classifier_arb_subsets.json")


# %%
# Print
print("finished")

# %%
