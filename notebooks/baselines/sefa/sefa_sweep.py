# %%
from __future__ import annotations

import copy
import os
from typing import Any

import lightning as pl
import mydatasets
import numpy as np
import sefalib
import tensordict as thd
import torch as th
import torch.utils.data as th_data

# %%
data_name: str = "gas"
model_name: str = "ours"
configs_sequence: str = "123"
num_repeats: int = 5

# %%
data = th.load(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "sefa", f"{data_name}.pt"
    ),
    weights_only=False,
)
tdata: thd.TensorDict = data["train"]
vdata: thd.TensorDict = data["valid"]
tstdata: thd.TensorDict = data["test"]
dataset_dict: dict[str, Any] = data["dataset_dict"]

# %%
postfix: str = "cdf" if model_name == "ours" else "std"
tdata_tensor_dataset = th_data.TensorDataset(
    tdata[f"xs_{postfix}"], tdata["ys"], tdata["ms"]
)
vdata_tensor_dataset = th_data.TensorDataset(
    vdata[f"xs_{postfix}"], vdata["ys"], vdata["ms"]
)
tstdata_tensor_dataset = th_data.TensorDataset(
    tstdata[f"xs_{postfix}"], tstdata["ys"], tstdata["ms"]
)

# %%
path = os.path.join(
    "experiments", "tuning", "tmp", data_name, f"{model_name}{configs_sequence}"
)
os.makedirs(path, exist_ok=True)

# %%
plf = pl.Fabric(accelerator="cpu")

# %%
# Get predictive metric function.
metric_f = sefalib.experiments.metrics_dict.metrics_dict[dataset_dict["metric"]]


# %%
# Construct the dictionary based on hyperparameters and dataset information.
def get_aucs(config):
    aucs = []
    for key in dataset_dict.keys():
        config[key] = dataset_dict[key]

    for repeat in range(1, num_repeats + 1):
        print(f"\n\nRepeat {repeat} out of {num_repeats}")
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))

        # Set the seed for consistency.
        seed = 1690 * repeat + 241
        np.random.seed(seed)
        th.manual_seed(seed)

        # Setup and train the model.
        model = sefalib.models.models_dict.models_dict[model_name](config).to(
            plf.device
        )
        model.fit(tdata_tensor_dataset, vdata_tensor_dataset, path, metric_f)
        aucs.append(th.load(os.path.join(path, "val_auc.pt")))

    aucs = np.array(aucs)
    return aucs


# %%
# Create dictionary of aucs, this is the random search.
hyperparams_dict = sefalib.experiments.tuning.trial_hyperparams.trial_hyperparams[
    model_name
][configs_sequence]
hyperparams_dict = copy.deepcopy(hyperparams_dict)
# for _k in hyperparams_dict:
#     hyperparams_dict[_k]["batchsize"] =
hyperparams_dict_no_dataset = copy.deepcopy(hyperparams_dict)
aucs_dict = {}
for key in hyperparams_dict.keys():
    aucs_dict[key] = get_aucs(hyperparams_dict[key])

# %%
# Print the results.
print("\n\n\nTuning resuts:\n")
best_mean = 0
for key in aucs_dict.keys():
    aucs = aucs_dict[key]
    mean = np.mean(aucs)
    std_err = np.std(aucs) / np.sqrt(len(aucs))
    print(f"{key}: {mean:.3f} +- {std_err:.3e}")
    if mean > best_mean:
        best_mean = mean
        best_std_err = std_err
        best_key = key
print(f"\n\nBest: {key}, {best_mean:.3f} +- {best_std_err:.3e}\n")
print(hyperparams_dict_no_dataset[best_key])

# %%
# Save the results in a text file to be used later.
path = os.path.join("experiments", "tuning", "results", data_name)
os.makedirs(path, exist_ok=True)

with open(os.path.join(path, f"{model_name}{configs_sequence}.txt"), "w") as f:
    for key in aucs_dict.keys():
        aucs = aucs_dict[key]
        mean = np.mean(aucs)
        std_err = np.std(aucs) / np.sqrt(len(aucs))
        f.write(f"{key} AUC: {mean:.3f} +- {std_err:.3e}\n")
    for key in aucs_dict.keys():
        f.write(f"\n\n{key} = ")
        f.write("{\n")
        for k in hyperparams_dict_no_dataset[key].keys():
            f.write(f'  "{k}": {hyperparams_dict_no_dataset[key][k]},\n')
        f.write("}")
    f.close()

# Delete the checkpoints from the tuning.
path = os.path.join(
    "experiments", "tuning", "tmp", data_name, f"{model_name}{configs_sequence}"
)
for file in os.listdir(path):
    os.remove(os.path.join(path, file))
os.rmdir(path)

# %%
