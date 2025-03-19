import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DefaultDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_parameters):
        features, labels = torch.tensor(data[0]), torch.tensor(data[1])
        self.data = torch.cat([features.float(), labels.float().unsqueeze(1)], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data


class DefaultCollateClass(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        if isinstance(batch[0], tuple):
            imgs = torch.cat([d[0].unsqueeze(0) for d in batch], dim=0)
            labels = torch.tensor([d[1] for d in batch]).unsqueeze(1)
            data = torch.cat([imgs.view(len(imgs), -1), labels.float()], dim=-1)
        else:
            data = torch.cat([d.unsqueeze(0) for d in batch], dim=0)

        return {"batch": data}


class VAECollateClass(object):
    def __init__(self, is_test):
        if is_test:
            self.g_cpu = torch.Generator()
            self.g_cpu.manual_seed(42)
            self.rng = default_rng(seed=420)
        else:
            self.g_cpu = None
            self.rng = default_rng()

    def __call__(self, batch):
        if isinstance(batch[0], tuple):
            imgs = torch.cat([d[0].unsqueeze(0) for d in batch], dim=0)
            labels = torch.tensor([d[1] for d in batch]).unsqueeze(1)
            data = torch.cat([imgs.view(len(imgs), -1), labels.float()], dim=-1)
        else:
            data = torch.cat([d.unsqueeze(0) for d in batch], dim=0)
        # n_observed is 0 to n_features+1 - 1
        observed = torch.randint(
            low=0, high=data.shape[1], generator=self.g_cpu, size=(data.shape[0],)
        )
        observed_mask = torch.zeros(data.shape)  # not all observed
        # nan_mask is 1 if original feature was unobserved to begin with
        nan_mask = torch.isnan(data).float()
        for idx in range(len(data)):
            if observed[idx] == 0:
                continue
            observed_mask[
                idx, self.rng.choice(data.shape[1], (observed[idx],), replace=False)
            ] = 1.0
        # unobserved_mask = 1 for unobserved nan features, a subset of observed features
        unobserved_mask = torch.max(
            (1.0 - observed_mask), nan_mask
        )  # either nan or subset of observed
        return {"batch": data, "mask": unobserved_mask}
