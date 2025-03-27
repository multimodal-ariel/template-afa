import json
import os

import numpy as np
import torch


def generate_random_feature_mask(batch, mask_type="none"):
    observed = torch.randint(low=0, high=batch.shape[1], size=(batch.shape[0],))
    observed_mask = torch.zeros_like(batch)
    nan_mask = torch.isnan(batch).float()
    for idx in range(len(batch)):
        if observed[idx] == 0:
            continue
        # label is never observed
        observed_mask[
            idx,
            np.random.choice(batch.shape[1] - 1, (observed[idx],), replace=False),
        ] = 1.0
    unobserved_mask = 1.0 - observed_mask
    if mask_type == "full":
        unobserved_mask[:, :-1] = 0.0
    unobserved_mask = torch.max(unobserved_mask, nan_mask)
    return unobserved_mask


def mask_batch(batch, mask):
    """
    Copy batch of objects and zero unobserved features.
    """
    observed_batch = batch.clone()
    observed_batch[mask == 1.0] = 0
    return observed_batch


def get_real_cat_features(data_parameters, add_label):

    n_features = data_parameters["n_features"]
    real_features = [
        idx
        for idx in range(n_features)
        if str(idx) not in data_parameters["categorical_classes"]
    ]
    cat_features = [
        idx
        for idx in range(n_features)
        if str(idx) in data_parameters["categorical_classes"]
    ]
    cat_categories = [
        data_parameters["categorical_classes"][str(cat)]
        for idx, cat in enumerate(cat_features)
    ]
    if add_label:
        if data_parameters["n_classes"] > 1:
            cat_features.append(n_features)
            cat_categories.append(data_parameters["n_classes"])
        else:
            real_features.append(n_features)
    return real_features, cat_features, cat_categories


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


def load_data_parameters(dataset):
    import difalib

    if os.path.exists(f"references/info_{dataset}.json"):
        return open_json("references/info_{}.json".format(dataset))
    data = difalib.models.data_utils.get_aaco_data(dataset)
    n_covs: int = data["train"][0].shape[1]
    n_labels: int = len(np.unique(data["train"][1]))
    return {
        "n_features": n_covs,
        "n_classes": n_labels,
        "categorical_classes": dict(),
        "shape": None,
    }
