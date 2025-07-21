"""XGBoost model that is used as a constant model for testing orders only.
The model assumes it receives torch tensors as input and has to output
them as well.

We use the XGBClassifier with 200 trees, max depth 4, 8 early stopping rounds.
The model works with missing values by converting them to np.nan. It trains by
uniformly subsamping the dataset as many times as we choose (10 usually),
concatenating them all and training on this.

For more on XGBoost see:
https://xgboost.readthedocs.io/en/stable/python/python_api.html
"""

import os.path as osp

import numpy as np
import torch
from xgboost import XGBClassifier


class XGBoostPredictor:
    """Assumes it receives torch tensors as input and output."""

    def __init__(self, n_estimators=200, max_depth=4, early_stopping_rounds=8):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            early_stopping_rounds=early_stopping_rounds,
        )

    def save(self, path):
        self.model.save_model(osp.join(path, "xgb_model.json"))

    def load(self, path):
        self.model.load_model(osp.join(path, "xgb_model.json"))

    def fit(self, train_set, val_set, path=None, num_subsamples=10):
        X_train = train_set[0].detach().cpu().numpy()
        y_train = train_set[1].detach().cpu().numpy()
        M_train = train_set[2].detach().cpu().numpy()
        X_val = val_set[0].detach().cpu().numpy()
        y_val = val_set[1].detach().cpu().numpy()
        M_val = val_set[2].detach().cpu().numpy()
        X_train_sub = [X_train]
        y_train_sub = [y_train]
        X_val_sub = [X_val]
        y_val_sub = [y_val]
        for _ in range(num_subsamples):
            # Train set.
            M_sub = (
                np.random.uniform(size=M_train.shape)
                > np.random.uniform(size=(M_train.shape[0], 1))
            ).astype(float) * M_train
            X_sub = np.where(M_sub, X_train, np.nan)
            X_train_sub.append(X_sub)
            y_train_sub.append(y_train)
            # Val set.
            M_sub = (
                np.random.uniform(size=M_val.shape)
                > np.random.uniform(size=(M_val.shape[0], 1))
            ).astype(float) * M_val
            X_sub = np.where(M_sub, X_val, np.nan)
            X_val_sub.append(X_sub)
            y_val_sub.append(y_val)
        X_train_sub = np.concatenate(X_train_sub, axis=0)
        y_train_sub = np.concatenate(y_train_sub, axis=0)
        X_val_sub = np.concatenate(X_val_sub, axis=0)
        y_val_sub = np.concatenate(y_val_sub, axis=0)
        self.model.fit(
            X_train_sub, y_train_sub, eval_set=[(X_val_sub, y_val_sub)], verbose=False
        )
        if path is not None:
            self.save(path)

    def predict(self, x, mask):
        # We return class probabilities as a tensor on the same device.
        x = np.where(mask.detach().cpu().numpy(), x.detach().cpu().numpy(), np.nan)
        return torch.tensor(self.model.predict_proba(x)).float().to(mask.device)
