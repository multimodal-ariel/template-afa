"""Two preprocessing shared modules:
1. Standardizer: Takes data and makes it mean 0 and std 1.
2. Vectorised Empirical CDF calculator, used to preprocess data specifically
for our model in accordance with https://arxiv.org/abs/1804.06216.
We also include a function to preprocess and save general data, this avoids
repeated code in the dataset scripts.
"""

import os
from typing import Any, Optional

import torch as th


class Standardizer:
    """Standardizes torch tensors to be mean 0 and std 1."""

    def __init__(self):
        pass

    def fit(self, x, mask=None):
        if mask is None:
            self.mean = th.mean(x, dim=0)
            self.std = th.std(x, dim=0)
        else:
            self.mean = th.sum(x * mask, dim=0) / th.sum(mask, dim=0)
            # Bessel's correction.
            var = th.sum(((x - self.mean) ** 2) * mask, dim=0) / (
                th.sum(mask, dim=0) - 1.0
            )
            self.std = var**0.5

    def transform(self, x, mask=None):
        with th.no_grad():
            x = (x - self.mean) / self.std
            if mask is not None:
                x = x * mask
            return x


class VectorEmpiricalCDF:
    """A torch vectorised empirical cdf calculator.
    We calculate by finding the locations of the quantiles, and then we
    linearly interpolate a general point. Finally we apply a Standard Normal
    inverse CDF to get the processed values.
    args:
      num_bins: int, the number of bins to use for the empirical cdf.
      size_normal: float, the standard deviation of the normal noise to add to the
                   data. This is added to prevent the cdf from being a step if
                   continuous values only fall into a finite set.
      ratio_uniform: float, the ratio of uniform noise to add to the data. We add
                    points sampled uniformly in the range of the data to prevent,
                    the cdf from losing some of the spatial information.
    """

    def __init__(self, num_bins=200, size_normal=1e-5, ratio_uniform=0.1):
        self.num_bins = num_bins
        self.size_normal = size_normal
        self.ratio_uniform = ratio_uniform

    def linspace_batched(self, start, stop, steps):
        ints = (th.arange(steps)).unsqueeze(0)
        dx = ((stop - start) / (steps - 1.0)).unsqueeze(-1)
        return start.unsqueeze(-1) + dx * ints

    def fit(self, x, mask=None):
        # If we have missing values we sample from the real data to fill them in.
        # This is acceptable since the CDFs are used to find quantiles, and each
        # feature is treated independently.
        # NOTE this assumes there are at least some non-missing values. If some
        # features are all missing, then we will not be able to fill them in.
        if mask is not None:
            for f in range(x.shape[-1]):
                m_tmp = mask[:, f]
                real_x = x[:, f][th.where(m_tmp)[0]]
                sampled_real_ids = th.multinomial(
                    th.ones(real_x.shape[0]), x.shape[0], replacement=True
                )
                sampled_real_x = real_x[sampled_real_ids]
                x[:, f] = m_tmp * x[:, f] + (1 - m_tmp) * sampled_real_x

        # Add normal noise.
        x = x + th.randn(size=x.shape) * self.size_normal

        # Concatenate uniform data.
        min = th.min(x, dim=0)[0]
        max = th.max(x, dim=0)[0]
        uniform_data = (max - min) * th.rand(
            size=(int(self.ratio_uniform * x.shape[0]), x.shape[-1])
        ) + min
        x = th.cat([x, uniform_data], dim=0)

        # Sort the data and find the quantiles.
        x = th.sort(x, dim=0)[0]
        batchsize = x.shape[0]
        num_features = x.shape[1]

        # We start by looking at equally spaced quantiles. Equally spaced cdfs looks
        # in detail at regions where CDF changes quickly.
        # Step separates train set into that num_bins equally spaced bins.
        step = int(batchsize / self.num_bins)

        # These are the quantile values, i.e. the first has to be the minimum,
        # and the last has to be the maximum. Carried out in a batched way.
        self.s = th.empty((num_features, self.num_bins))
        self.s[:, 0] = x[0, :]
        self.s[:, -1] = x[-1, :]

        # These are the cdf values at the quantile values, i.e. range from 0.0 to 1.0.
        self.cdf = th.empty((num_features, self.num_bins))
        # This is an unbiased estimate of the min and max values drawn uniformly.
        # Rather than 0.0 and 1.0.
        self.cdf[:, 0] = 1 / (batchsize + 1)
        self.cdf[:, -1] = batchsize / (batchsize + 1)

        for bin in range(1, self.num_bins - 1):
            self.s[:, bin] = x[step * bin, :]  # What x value is at this quantile.
            self.cdf[:, bin] = (
                step * bin / batchsize
            )  # What fraction of the data are we.

        # We then look at bins that are equally spaced in the x axis, this will
        # look in closer details at regions where the CDF changes slowly, which were
        # not previously captured.
        equally_spaced = self.linspace_batched(x[0, :], x[-1, :], self.num_bins)
        ids = th.searchsorted(
            (x.T).contiguous(), equally_spaced.contiguous(), right=True
        )
        cdf_x = ids / batchsize
        s_x = th.gather(
            input=x.T, index=th.clamp(ids, min=0, max=batchsize - 1), dim=-1
        )
        s_x[:, 0] = x[0, :]
        s_x[:, -1] = x[-1, :]
        cdf_x[:, 0] = 1 / (batchsize + 1)
        cdf_x[:, -1] = batchsize / (batchsize + 1)

        # Finally we merge the two, sort  and remove the first and last since they
        # are duplicates.
        self.s = th.cat([self.s, s_x], dim=-1)
        self.cdf = th.cat([self.cdf, cdf_x], dim=-1)
        self.s = th.sort(self.s, dim=-1)[0][:, 1:-1]
        self.cdf = th.sort(self.cdf, dim=-1)[0][:, 1:-1]

        # m is the constant in y = mx + c
        self.m = (self.cdf[:, 1:] - self.cdf[:, :-1]) / (self.s[:, 1:] - self.s[:, :-1])
        self.m = th.cat(
            [th.zeros(self.m.shape[0], 1), self.m, th.zeros(self.m.shape[0], 1)],
            dim=-1,
        )

    def empirical_cdf(self, x):
        ids = th.searchsorted(self.s.contiguous(), (x.T).contiguous(), right=True)
        m_ = th.gather(input=self.m, index=ids, dim=-1).T
        # Shift ids back to correctly select the cdf and x_ values.
        clamped_ids = th.clamp(ids - 1, max=self.cdf.shape[-1] - 1, min=0)
        c_ = th.gather(input=self.cdf, index=clamped_ids, dim=-1).T
        x_ = th.gather(input=self.s, index=clamped_ids, dim=-1).T
        return m_ * (x - x_) + c_

    def transform(self, x, mask=None):
        with th.no_grad():
            x = th.clamp(self.empirical_cdf(x), min=1e-7, max=1.0 - 1e-7)
            x = 2**0.5 * th.erfinv(2 * x - 1)
            if mask is not None:
                x = x * mask
            return x


def preprocess_and_save_data(
    path,
    dataset_dict,
    train_size,
    val_size,
    X,
    y,
    M=None,
    shuffle=False,
    num_bins=100,
    size_normal=0.0,
    ratio_uniform=0.0,
):
    # Common function to preprocess and save all data to avoid repeated code.

    # Shuffle (if required) and split into train, val, test.
    if shuffle:
        shuffle_ids = th.randperm(X.shape[0])
        X = X[shuffle_ids]
        y = y[shuffle_ids]

    X_train = X[:train_size]
    X_val = X[train_size : train_size + val_size]
    X_test = X[train_size + val_size :]

    y_train = y[:train_size]
    y_val = y[train_size : train_size + val_size]
    y_test = y[train_size + val_size :]

    if M is not None:
        M = M[shuffle_ids] if shuffle else M
        M_train = M[:train_size]
        M_val = M[train_size : train_size + val_size]
        M_test = M[train_size + val_size :]
    else:
        M_train = th.ones_like(X_train)
        M_val = th.ones_like(X_val)
        M_test = th.ones_like(X_test)

    # Split the continuous and categorical features.
    num_con_features = dataset_dict["num_con_features"]
    X_con_train = X_train[:, :num_con_features]
    X_con_val = X_val[:, :num_con_features]
    X_con_test = X_test[:, :num_con_features]

    if M is not None:
        M_con_train = M_train[:, :num_con_features]
        M_con_val = M_val[:, :num_con_features]
        M_con_test = M_test[:, :num_con_features]
    else:
        M_con_train = None
        M_con_val = None
        M_con_test = None

    if dataset_dict["num_cat_features"] == 0:
        X_cat_train = th.empty(size=(X_con_train.shape[0], 0))
        X_cat_val = th.empty(size=(X_con_val.shape[0], 0))
        X_cat_test = th.empty(size=(X_con_test.shape[0], 0))
    else:
        X_cat_train = X_train[:, num_con_features:]
        X_cat_val = X_val[:, num_con_features:]
        X_cat_test = X_test[:, num_con_features:]

    # Preprocess the continuous features.
    # CDF normalize the continuous features.
    empirical_cdf = VectorEmpiricalCDF(
        num_bins=num_bins, size_normal=size_normal, ratio_uniform=ratio_uniform
    )
    empirical_cdf.fit(X_con_train, M_con_train)
    X_con_train_cdf = empirical_cdf.transform(X_con_train, M_con_train)
    X_con_val_cdf = empirical_cdf.transform(X_con_val, M_con_val)
    X_con_test_cdf = empirical_cdf.transform(X_con_test, M_con_test)

    # Make mean 0 and standard deviation 0.
    standardizer = Standardizer()
    standardizer.fit(X_con_train, M_con_train)
    X_con_train_std = standardizer.transform(X_con_train, M_con_train)
    X_con_val_std = standardizer.transform(X_con_val, M_con_val)
    X_con_test_std = standardizer.transform(X_con_test, M_con_test)

    X_train_cdf = th.cat((X_con_train_cdf.float(), X_cat_train.float()), dim=1)
    X_val_cdf = th.cat((X_con_val_cdf.float(), X_cat_val.float()), dim=1)
    X_test_cdf = th.cat((X_con_test_cdf.float(), X_cat_test.float()), dim=1)

    X_train_std = th.cat((X_con_train_std.float(), X_cat_train.float()), dim=1)
    X_val_std = th.cat((X_con_val_std.float(), X_cat_val.float()), dim=1)
    X_test_std = th.cat((X_con_test_std.float(), X_cat_test.float()), dim=1)

    th.save(X_train_cdf, os.path.join(path, "X_train_cdf.pt"))
    th.save(X_val_cdf, os.path.join(path, "X_val_cdf.pt"))
    th.save(X_test_cdf, os.path.join(path, "X_test_cdf.pt"))

    th.save(X_train_std, os.path.join(path, "X_train_std.pt"))
    th.save(X_val_std, os.path.join(path, "X_val_std.pt"))
    th.save(X_test_std, os.path.join(path, "X_test_std.pt"))

    th.save(y_train.long(), os.path.join(path, "y_train.pt"))
    th.save(y_val.long(), os.path.join(path, "y_val.pt"))
    th.save(y_test.long(), os.path.join(path, "y_test.pt"))

    th.save(M_train.float(), os.path.join(path, "M_train.pt"))
    th.save(M_val.float(), os.path.join(path, "M_val.pt"))
    th.save(M_test.float(), os.path.join(path, "M_test.pt"))

    th.save(dataset_dict, os.path.join(path, "dataset_dict.pt"))
