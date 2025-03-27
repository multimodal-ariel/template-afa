import os
import pickle
import random

import mydatasets.aaco
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as tt

from .dataset import (
    DatasetFromSubset,
    DefaultCollateClass,
    DefaultDataset,
    VAECollateClass,
)
from .utils import dump_json, load_data_parameters


def gen_cube(n_features=20, data_points=20000, sigma=0.1, seed=123):
    assert n_features >= 10, "cube data have >= 10 num of features"
    np.random.seed(seed)
    clean_points = np.random.binomial(1, 0.5, (data_points, 3))
    labels = np.dot(clean_points, np.array([1, 2, 4]))
    points = clean_points + np.random.normal(0, sigma, (data_points, 3))
    features = np.random.rand(data_points, n_features)
    for i in range(data_points):
        offset = labels[i]
        for j in range(3):
            features[i, offset + j] = points[i, j]
    return (features,)


def get_propulsion_dataset():
    pass
    # data/UCI_CBM_Dataset


def normalize_features(features, real_features):
    updated_features = features.copy()
    for col_idx in real_features:
        # scaler = preprocessing.StandardScaler().fit(features[:, col_idx : col_idx + 1])
        # updated_features[:, col_idx : col_idx + 1] = scaler.transform(
        #    features[:, col_idx : col_idx + 1]
        # )
        not_nan = ~np.isnan(features[:, col_idx])
        mean, std = np.mean(features[not_nan, col_idx]), np.std(
            features[not_nan, col_idx]
        )
        updated_features[not_nan, col_idx] = (
            updated_features[not_nan, col_idx] - mean
        ) / (1e-8 + std)
    return updated_features


def train_test_split(X, y, test_size, random_state):
    n = len(X)
    n_train = n - int(n * test_size)
    indices = [idx for idx in range(n)]
    random.Random(random_state).shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def split_data(X, y, fixed_test=None):
    if fixed_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test = X[:-fixed_test, :], X[-fixed_test:, :]
        y_train, y_test = y[:-fixed_test], y[-fixed_test:]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    return {
        "train": [X_train, y_train],
        "valid": [X_valid, y_valid],
        "test": [X_test, y_test],
    }


def dump_info(name, n_features, n_classes, categorical_classes, shape=None):
    output = {
        "n_features": n_features,
        "n_classes": n_classes,
        "categorical_classes": categorical_classes,
        "shape": shape,
    }
    if not os.path.exists("references/info_{}.json".format(name)):
        dump_json("references/info_{}.json".format(name), output)


def get_wine_dataset():
    data = pd.read_csv("data/winequality-white.csv", delimiter=";").to_numpy()
    features, labels = data[:, :-1], data[:, -1]
    # min_, max_ = np.min(labels), np.max(labels)
    # labels = (labels - min_) / (max_ - min_)
    mean, std = np.mean(labels), np.std(labels)
    labels = (labels - mean) / (1e-8 + std)
    n_features, categorical_classes, n_classes = len(features[0]), {}, 1
    real_features = [idx for idx in range(n_features)]
    dump_info("wine", n_features, n_classes, categorical_classes)
    features = normalize_features(features, real_features)
    return split_data(features, labels)


def get_grid_dataset():
    data = pd.read_csv("data/Data_for_UCI_named.csv")
    data.loc[data["stabf"] == "stable", "stabf"] = 1
    data.loc[data["stabf"] == "unstable", "stabf"] = 0
    data = data.to_numpy().astype("float64")
    features, labels = data[:, :-2], data[:, -1]
    n_features, categorical_classes, n_classes = len(features[0]), {}, 2
    real_features = [idx for idx in range(n_features)]
    dump_info("grid", n_features, n_classes, categorical_classes)
    features = normalize_features(features, real_features)
    return split_data(features, labels)


def get_gas_dataset():
    data = pd.read_csv("data/HT_Sensor_dataset.dat", delimiter=r"\s+")
    meta_data = pd.read_csv("data/HT_Sensor_metadata.dat", delimiter=r"\s+")
    ids = [x for x in meta_data["id"]]
    f = {"banana": 0, "wine": 1, "background": 2}
    classes = [f[x] for x in meta_data["class"]]
    meta_data = {id_: x for (id_, x) in zip(ids, classes)}
    data["id"].replace(meta_data, inplace=True)
    data = data.to_numpy()
    features, labels = data[:, 2:], data[:, 0]
    n_features, categorical_classes, n_classes = len(features[0]), {}, 3
    real_features = [idx for idx in range(n_features)]
    dump_info("gas", n_features, n_classes, categorical_classes)
    features = normalize_features(features, real_features)
    return split_data(features, labels)


def get_parkinsons_dataset():
    data = pd.read_csv("data/parkinsons_updrs.data", delimiter=",")
    data = data.to_numpy()
    features, labels = data[:, -16:], data[:, -17]
    features = features.astype("float64")
    labels = labels.astype("float64")
    # min_, max_ = np.min(labels), np.max(labels)
    # labels = (labels - min_) / (max_ - min_)
    mean, std = np.mean(labels), np.std(labels)
    print(mean, std)
    labels = (labels - mean) / (1e-8 + std)
    n_features, categorical_classes, n_classes = len(features[0]), {}, 1
    real_features = [idx for idx in range(n_features)]
    dump_info("parkinsons", n_features, n_classes, categorical_classes)
    features = normalize_features(features, real_features)
    return split_data(features, labels)


def get_fashionmnist_dataset():
    stats = ((0.2860), (0.3530))
    transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    train_dataset = torchvision.datasets.FashionMNIST(
        root="data/", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root="data/", train=False, download=True, transform=transform
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) - 6000, 6000],
        generator=torch.Generator().manual_seed(42),
    )
    n_features, categorical_classes, n_classes = 28 * 28, {}, 10
    dump_info("fashionmnist", n_features, n_classes, categorical_classes, [1, 28, 28])

    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def get_cifar_dataset_aug(params):
    data_name = params.data
    if data_name == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        n_features, categorical_classes, n_classes = 3 * 32 * 32, {}, 10
        dump_info("cifar10", n_features, n_classes, categorical_classes, [3, 32, 32])

    elif data_name == "cifar100":
        stats = ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        n_features, categorical_classes, n_classes = 3 * 32 * 32, {}, 100
        dump_info("cifar100", n_features, n_classes, categorical_classes, [3, 32, 32])

    train_transform = tt.Compose(
        [
            tt.RandomCrop(32, padding=4, padding_mode="reflect"),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True),
        ]
    )
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    if not params.augmentation:
        train_transform = test_transform
    if data_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="data/", train=True, download=True
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root="data/", train=False, download=True, transform=test_transform
        )
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root="data/", train=True, download=True
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="data/", train=False, download=True, transform=test_transform
        )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) - 5000, 5000],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = DatasetFromSubset(train_dataset, train_transform)
    valid_dataset = DatasetFromSubset(valid_dataset, test_transform)
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def get_cifar100_dataset(params):
    return get_cifar_dataset_aug(params)

    train_dataset = torchvision.datasets.CIFAR100(
        root="data/", train=True, download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="data/", train=False, download=True
    )
    train_X, train_y = (train_dataset.data, train_dataset.targets)
    test_X, test_y = test_dataset.data, test_dataset.targets
    X, y = np.concatenate([train_X, test_X], axis=0), np.concatenate(
        [train_y, test_y], axis=0
    )
    X = torch.tensor(X.transpose((0, 3, 1, 2))).contiguous().float()  # BCHW
    mean, std = (
        torch.mean(X, dim=(0, 2, 3), keepdim=True),
        torch.std(X, unbiased=True, dim=(0, 2, 3), keepdim=True),
    )
    X = (X - mean) / std
    X = X.view(-1, 3 * 32 * 32).float()
    X = X.numpy()
    n_features, categorical_classes, n_classes = len(X[0]), {}, 100
    dump_info("cifar100", n_features, n_classes, categorical_classes, [3, 32, 32])
    return split_data(X, y, fixed_test=10000)


def get_cifar10_dataset(params):
    return get_cifar_dataset_aug(params)
    train_dataset = torchvision.datasets.CIFAR10(
        root="data/", train=True, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/", train=False, download=True
    )
    train_X, train_y = (train_dataset.data, train_dataset.targets)
    test_X, test_y = test_dataset.data, test_dataset.targets
    X, y = np.concatenate([train_X, test_X], axis=0), np.concatenate(
        [train_y, test_y], axis=0
    )
    X = torch.tensor(X.transpose((0, 3, 1, 2))).contiguous().float()  # BCHW
    mean, std = (
        torch.mean(X, dim=(0, 2, 3), keepdim=True),
        torch.std(X, unbiased=True, dim=(0, 2, 3), keepdim=True),
    )
    X = (X - mean) / std
    X = X.view(-1, 3 * 32 * 32).float()
    X = X.numpy()
    n_features, categorical_classes, n_classes = len(X[0]), {}, 10
    dump_info("cifar10", n_features, n_classes, categorical_classes, [3, 32, 32])
    return split_data(X, y, fixed_test=10000)


def get_stl_dataset(params):
    stats = ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    n_features, categorical_classes, n_classes = 3 * 96 * 96, {}, 10
    dump_info("stl", n_features, n_classes, categorical_classes, [3, 96, 96])

    train_transform = tt.Compose(
        [
            tt.RandomCrop(96, padding=4, padding_mode="reflect"),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True),
        ]
    )
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    if not params.augmentation:
        train_transform = test_transform

    train_dataset = torchvision.datasets.STL10(
        root="data/", split="train", download=True
    )

    test_dataset = torchvision.datasets.STL10(root="data/", split="test", download=True)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [10000, 1000, 2000],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset = DatasetFromSubset(train_dataset, train_transform)
    valid_dataset = DatasetFromSubset(valid_dataset, test_transform)
    test_dataset = DatasetFromSubset(test_dataset, test_transform)
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def get_food_dataset(params):
    stats = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    n_features, categorical_classes, n_classes = 3 * 32 * 32, {}, 101
    dump_info("food", n_features, n_classes, categorical_classes, [3, 32, 32])

    train_transform = tt.Compose(
        [
            tt.RandomCrop(32, padding=4, padding_mode="reflect"),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True),
        ]
    )
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    if not params.augmentation:
        train_transform = test_transform

    train_dataset = torchvision.datasets.Food101(
        root="data/", split="train", download=True
    )

    test_dataset = torchvision.datasets.Food101(
        root="data/", split="test", download=True, transform=test_transform
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) - 8000, 8000],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset = DatasetFromSubset(train_dataset, train_transform)
    valid_dataset = DatasetFromSubset(valid_dataset, test_transform)
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def get_svhn_dataset(params):
    stats = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    n_features, categorical_classes, n_classes = 3 * 32 * 32, {}, 10
    dump_info("svhn", n_features, n_classes, categorical_classes, [3, 32, 32])

    train_transform = tt.Compose(
        [
            tt.RandomCrop(32, padding=4, padding_mode="reflect"),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True),
        ]
    )
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    if not params.augmentation:
        train_transform = test_transform

    train_dataset = torchvision.datasets.SVHN(
        root="data/", split="train", download=True
    )

    test_dataset = torchvision.datasets.SVHN(
        root="data/", split="test", download=True, transform=test_transform
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) - 8000, 8000],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = DatasetFromSubset(train_dataset, train_transform)
    valid_dataset = DatasetFromSubset(valid_dataset, test_transform)
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def get_mnist_dataset(subsample=False):
    train_dataset = torchvision.datasets.MNIST(root="data/", train=True, download=True)

    test_dataset = torchvision.datasets.MNIST(root="data/", train=False, download=True)
    train_X, train_y = train_dataset.data.numpy(), train_dataset.targets.numpy()
    test_X, test_y = test_dataset.data.numpy(), test_dataset.targets.numpy()
    X, y = np.concatenate([train_X, test_X], axis=0), np.concatenate(
        [train_y, test_y], axis=0
    )
    X = torch.tensor(X).view(-1, 28 * 28).float()
    if subsample:
        X = X.view(-1, 1, 28, 28)
        X = torch.nn.functional.interpolate(X, (16, 16))
        X = X.view(-1, 256)
    mean, std = torch.mean(X, dim=(0, 1), keepdim=True), torch.std(
        X, dim=(0, 1), unbiased=True, keepdim=True
    )
    X = (X - mean) / std
    X = X.numpy()
    n_features, categorical_classes, n_classes = len(X[0]), {}, 10
    dump_info(
        "mnist16" if subsample else "mnist",
        n_features,
        n_classes,
        categorical_classes,
        [1, 16, 16] if subsample else [1, 28, 28],
    )
    return split_data(X, y, fixed_test=10000)


def get_adult_dataset():
    if os.path.exists("data/adult.pkl"):
        with open("data/adult.pkl", "rb") as f:
            return pickle.load(f)

    train_df = pd.read_csv("data/adult/adult.data", header=None)
    test_df = pd.read_csv("data/adult/adult.test", skiprows=1, header=None)
    df = pd.concat([train_df, test_df])
    x = df.to_numpy()
    N, d = x.shape
    for row in range(N):
        x[row, -1] = x[row, -1].strip(".")
    mappers = {}
    categorical_classes = {}
    for col in range(d):
        if not isinstance(x[0, col], str):
            continue
        uniques = [d.strip() for d in np.unique(x[:, col]).tolist() if d.strip() != "?"]
        uniques = {d: idx for idx, d in enumerate(uniques)}
        uniques["?"] = np.nan
        mappers[col] = uniques
        if col != d - 1:
            categorical_classes[col] = len(uniques) - 1
        for row in range(N):
            x[row, col] = uniques[x[row, col].strip()]
    features, labels = x[:, :-1], x[:, -1]
    features = features.astype("float64")
    labels = labels.astype("float64")
    n_features, n_classes = len(features[0]), 2
    real_features = [idx for idx in range(n_features) if idx not in categorical_classes]
    features = normalize_features(features, real_features)
    data_dict = split_data(features, labels)
    data_dict["mappers"] = mappers
    dump_info("adult", n_features, n_classes, categorical_classes, None)
    # max 3 nan features
    with open("data/adult.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    return data_dict


def get_bank_dataset():
    if os.path.exists("data/bank.pkl"):
        with open("data/bank.pkl", "rb") as f:
            return pickle.load(f)

    df = pd.read_csv("data/bank-additional/bank-additional-full.csv", delimiter=";")
    df.drop("duration", inplace=True, axis=1)
    x = df.to_numpy()
    N, d = x.shape
    mappers = {}
    categorical_classes = {}
    for col in range(d):
        if not isinstance(x[0, col], str):
            continue
        uniques = [
            d.strip() for d in np.unique(x[:, col]).tolist() if d.strip() != "unknown"
        ]
        uniques = {d: idx for idx, d in enumerate(uniques)}
        uniques["unknown"] = np.nan
        mappers[col] = uniques
        if col != d - 1:
            categorical_classes[col] = len(uniques) - 1
        for row in range(N):
            x[row, col] = uniques[x[row, col].strip()]
    features, labels = x[:, :-1], x[:, -1]
    features = features.astype("float64")
    labels = labels.astype("float64")
    n_features, n_classes = len(features[0]), 2
    real_features = [idx for idx in range(n_features) if idx not in categorical_classes]
    features = normalize_features(features, real_features)
    data_dict = split_data(features, labels)
    data_dict["mappers"] = mappers
    dump_info("bank", n_features, n_classes, categorical_classes, None)
    # max 3 nan features
    with open("data/bank.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    return data_dict


def get_blog_dataset():
    test_df = pd.concat(
        [
            pd.read_csv("data/blog/{}".format(d), header=None)
            for d in os.listdir("data/blog/")
            if "blogData_test" in d
        ]
    )
    train_df = pd.read_csv("data/blog/blogData_train.csv", header=None)
    test_size = test_df.shape[0]
    data = pd.concat([train_df, test_df]).to_numpy()
    features, labels = data[:, :-1], data[:, -1]
    mean, std = np.mean(labels), np.std(labels)
    labels = (labels - mean) / (1e-8 + std)
    n_features, categorical_classes, n_classes = len(features[0]), {}, 1
    real_features = [idx for idx in range(n_features)]
    dump_info("blog", n_features, n_classes, categorical_classes)
    features = normalize_features(features, real_features)
    X, y = features, labels
    X_train, X_test = X[:-test_size, :], X[-test_size:, :]
    y_train, y_test = y[:-test_size], y[-test_size:]
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, y_test, test_size=0.33, random_state=42
    )
    return {
        "train": [X_train, y_train],
        "valid": [X_valid, y_valid],
        "test": [X_test, y_test],
    }


def get_youtube_dataset():
    pass


def get_fb_dataset():
    pass


def get_physionet_dataset(set_name):
    outcome_file = "data/physionet/Outcomes-{}.txt".format(set_name)
    mortality = {}
    with open(outcome_file, "r") as fp:
        lines = fp.readlines()
        lines = [line.strip("\n") for line in lines[1:]]
        for line in lines:
            words = line.split(",")
            recordid, death = int(words[0]), float(words[-1])
            mortality[recordid] = death
    datafolder = "data/physionet/set-{}/".format(set_name)
    json_data = []
    for f in os.listdir(datafolder):
        with open(os.path.join(datafolder, f), "r") as fp:
            lines = fp.readlines()
        lines = [line.strip("\n") for line in lines[1:]]
        data = {}
        for line in lines:
            field, value = line.split(",")[1:]
            value = float(value) if field != "RecordID" else int(value)
            if value >= 0:
                data[field] = value
        try:
            data["death"] = mortality[data["RecordID"]]
            json_data.append(data)
        except KeyError:
            continue
    df = pd.DataFrame(json_data)
    return df


def get_physio_mortality_dataset():
    if os.path.exists("data/physio-mortality.pkl"):
        with open("data/physio-mortality.pkl", "rb") as f:
            return pickle.load(f)

    df_a = get_physionet_dataset("a")
    df_b = get_physionet_dataset("b")
    df_c = get_physionet_dataset("c")
    df = pd.concat([df_c, df_b, df_a])
    df.drop("", inplace=True, axis=1)
    df.drop("RecordID", inplace=True, axis=1)
    cols = df.columns.tolist()
    cols.pop(cols.index("death"))
    cols.append("death")
    df = df[cols]
    x = df.to_numpy()
    N, d = x.shape
    mappers = {}
    categorical_classes = {}
    for col_idx, col in enumerate(cols):
        if col not in {"Gender", "ICUType"}:
            continue
        uniques = [d for d in np.unique(x[:, col_idx]).tolist() if d == d]
        uniques = {d: idx for idx, d in enumerate(uniques)}
        uniques[np.nan] = np.nan
        mappers[col] = uniques
        categorical_classes[col_idx] = len(uniques) - 1
        for row in range(N):
            if not np.isnan(x[row, col_idx]):
                x[row, col_idx] = uniques[x[row, col_idx]]
    features, labels = x[:, :-1], x[:, -1]
    nans = np.isnan(features).sum(axis=-1) >= 21
    features = features[~nans]
    labels = labels[~nans]
    features = features.astype("float64")
    labels = labels.astype("float64")
    n_features, n_classes = len(features[0]), 2
    real_features = [idx for idx in range(n_features) if idx not in categorical_classes]
    features = normalize_features(features, real_features)
    data_dict = split_data(features, labels)
    data_dict["mappers"] = mappers
    dump_info("physio-mortality", n_features, n_classes, categorical_classes, None)
    # max 3 nan features
    with open("data/physio-mortality.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    return data_dict


def get_mimic_dataset(set_name):
    outcome_file = (
        "../mimic3-benchmarks/data/in-hospital-mortality/{}/listfile.csv".format(
            set_name
        )
    )
    mortality = {}
    with open(outcome_file, "r") as fp:
        lines = fp.readlines()
        lines = [line.strip("\n") for line in lines[1:]]
        for line in lines:
            words = line.split(",")
            recordid, death = (words[0]), float(words[-1])
            mortality[recordid] = death
    datafolder = "../mimic3-benchmarks/data/in-hospital-mortality/{}/".format(set_name)
    json_data = []
    for f in mortality.keys():
        with open(os.path.join(datafolder, f), "r") as fp:
            lines = fp.readlines()
        lines = [line.strip("\n") for line in lines]
        header = {idx: col for idx, col in enumerate(lines[0].split(","))}
        data = {}
        for line in lines[1:]:
            values = line.split(",")
            for idx, val in enumerate(values):
                if val != "":
                    data[header[idx]] = val
        data["death"] = mortality[f]
        json_data.append(data)
    df = pd.DataFrame(json_data)
    return df


def get_in_hospital_mortality_dataset():
    if os.path.exists("data/in-hospital-mortality.pkl"):
        with open("data/in-hospital-mortality.pkl", "rb") as f:
            return pickle.load(f)

    df_a = get_mimic_dataset("test")
    df_b = get_mimic_dataset("train")
    df = pd.concat([df_a, df_b])
    df.drop("Hours", inplace=True, axis=1)
    cols = df.columns.tolist()
    cols.pop(cols.index("death"))
    cols.append("death")
    df = df[cols]
    mappers = {}
    categorical_classes = {}
    for col_idx, col in enumerate(cols):
        if not ("Glascow" in col and "total" not in col):
            continue
        uniques = [d for d in df[col].unique() if d == d]
        uniques = {d: idx for idx, d in enumerate(uniques)}
        uniques[np.nan] = np.nan
        mappers[col] = uniques
        categorical_classes[col_idx] = len(uniques) - 1
        df = df.replace({col: uniques})
    x = df.to_numpy()
    x = np.random.permutation(x)
    N, d = x.shape
    features, labels = x[:, :-1], x[:, -1]
    features = features.astype("float64")
    labels = labels.astype("float64")
    nans = np.isnan(features).sum(axis=-1) >= 7
    features = features[~nans]
    labels = labels[~nans]
    n_features, n_classes = len(features[0]), 2
    real_features = [idx for idx in range(n_features) if idx not in categorical_classes]
    features = normalize_features(features, real_features)
    data_dict = split_data(features, labels)
    data_dict["mappers"] = mappers
    dump_info(
        "in-hospital-mortality",
        n_features,
        n_classes,
        categorical_classes,
        None,
    )
    # max 3 nan features
    with open("data/in-hospital-mortality.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    return data_dict


def get_aaco_data(params_data_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    dname_psd_l: list[str] = params_data_name.split("/")
    to_normalize = True
    if len(dname_psd_l) > 2:
        _to_normalize_str = dname_psd_l[2].lower()
        if _to_normalize_str == "true" or _to_normalize_str != "false":
            to_normalize = True
        elif _to_normalize_str != "true" or _to_normalize_str == "false":
            to_normalize = False
        else:
            raise ValueError(
                f"fail to parse {dname_psd_l[2]} of str {params_data_name}"
            )
    tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(
        name=dname_psd_l[1], to_normalize=to_normalize
    )
    return {
        "train": (tdata["xs"].numpy(), tdata["ys"].numpy()),
        "valid": (vdata["xs"].numpy(), vdata["ys"].numpy()),
        "test": (tstdata["xs"].numpy(), tstdata["ys"].numpy()),
    }


def get_data(params):
    data_name = params.data
    if data_name.split("/")[0] == "load_aaco_data":
        return get_aaco_data(data_name)
    if data_name == "adult":
        return get_adult_dataset()
    if data_name == "bank":
        return get_bank_dataset()
    if data_name == "blog":
        return get_blog_dataset()
    if data_name == "youtube":
        return get_youtube_dataset()
    if data_name == "facebook":
        return get_fb_dataset()
    if data_name == "gas":
        return get_gas_dataset()
    if data_name == "grid":
        return get_grid_dataset()
    if data_name == "cifar10":
        return get_cifar10_dataset(params)
    if data_name == "cifar100":
        return get_cifar100_dataset(params)
    if data_name == "mnist16":
        return get_mnist_dataset(subsample=True)
    if data_name == "mushroom":
        pass
    if data_name == "stl":
        return get_stl_dataset(params)
    if data_name == "food":
        return get_food_dataset(params)
    if data_name == "svhn":
        return get_svhn_dataset(params)
    if data_name == "mnist":
        return get_mnist_dataset()
    if data_name == "fashionmnist":
        return get_fashionmnist_dataset()
    if data_name == "parkinsons":
        return get_parkinsons_dataset()
    if data_name == "wine":
        return get_wine_dataset()
    if data_name == "physio-mortality":
        return get_physio_mortality_dataset()
    if data_name == "in-hospital-mortality":
        return get_in_hospital_mortality_dataset()
    if data_name in {"cube", "cube-reg"}:
        with open("data/cube.pkl", "rb") as f:
            return pickle.load(f)

    # if not os.path.isfile("data/{}.pkl".format(model.params.data)):


def prepare_data(model):
    data_dict = get_data(model.params)
    dataset_classes = {}
    Dataset = dataset_classes.get(model.params.data, DefaultDataset)
    data_parameters = load_data_parameters(model.params.data)
    if not isinstance(data_dict["train"], list):
        model.trainset = data_dict["train"]
        model.testset = data_dict["test"]
        model.validset = data_dict["valid"]
        return data_parameters
    model.trainset = Dataset(data_dict["train"], data_parameters)
    model.validset = Dataset(data_dict["valid"], data_parameters)
    model.testset = Dataset(data_dict["test"], data_parameters)
    return data_parameters


def load_data(model):
    if model.params.problem == "vaeac":
        train_collate_fn = VAECollateClass(is_test=False)
        test_collate_fn = VAECollateClass(is_test=True)
    else:
        train_collate_fn = DefaultCollateClass()
        test_collate_fn = DefaultCollateClass()
    train_loader = torch.utils.data.DataLoader(
        model.trainset,
        collate_fn=train_collate_fn,
        batch_size=model.params.batch_size,
        num_workers=model.params.workers,
        drop_last=True,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        model.validset,
        collate_fn=test_collate_fn,
        batch_size=model.params.batch_size * 2,
        num_workers=model.params.workers,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        model.testset,
        collate_fn=test_collate_fn,
        batch_size=model.params.batch_size * 2,
        num_workers=model.params.workers,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
