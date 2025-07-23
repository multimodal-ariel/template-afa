"""Base Models for any acquisition model.

All models inherit from this class. This class is responsible for the main
training loop, saving/loading, and acquisition. We acquire by scoring each
feature based on the information we have, each individual model does that.
Then we choose the feature with the largest score that is available and that
we have not already acquired.
"""

import gc
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc as sklearn_auc
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .constants import cooldown, lr_factor, min_lr
from .standard_layers import ContinuousInput, MixedInput


class BaseModel(nn.Module):
    """Base acquisition model. Is able to predict y and update the mask.
    Models change how they predict and how they score feature acquisitions.
    Model parameter fitting also changes based on the specific model.
    """

    def __init__(self, config):
        super().__init__()
        # Input parameters.
        self.in_dim = 2 * config["num_con_features"] + config["num_cat_features"] * (
            config["most_categories"] + 1
        )
        if config["num_cat_features"] == 0:
            self.mixed_input = False
            self.input_layer = ContinuousInput()
        else:
            # Assumes we always have some continuous features.
            self.mixed_input = True
            self.input_layer = MixedInput(
                config["num_con_features"], config["most_categories"]
            )
        # Optimization parameters, some models (like RL) do not use these.
        self.epochs = config["epochs"] if "epochs" in config else None
        self.lr = config["lr"] if "lr" in config else None
        self.batchsize = config["batchsize"] if "batchsize" in config else None
        self.patience = config["patience"] if "patience" in config else None
        # Other parameters.
        self.num_features = config["num_con_features"] + config["num_cat_features"]
        self.max_dim = (
            config["max_dim"] if config["max_dim"] is not None else self.num_features
        )
        self.out_dim = config["out_dim"]
        self.use_fixed_order = False

    @property
    def device(self):
        return next(self.parameters()).device

    def clear_cache(self):
        if self.device == torch.device("cpu"):
            gc.collect()
        else:
            torch.cuda.empty_cache()

    def predict(self, x, mask):
        # Needs to predict the distribution over the output, not the logits.
        raise NotImplementedError

    def loss_func(self, x, y, mask, data_mask=None):
        raise NotImplementedError

    def load(self, path):
        self.load_state_dict(torch.load(osp.join(path, "best_model.pt")))
        self.eval()
        if self.use_fixed_order:
            self.fixed_order_scores = torch.load(
                osp.join(path, "fixed_order_scores.pt")
            ).to(self.device)

    def subsample_mask(self, mask):
        # Subsamples mask with cardinality chosen uniformly. Multiply by true mask
        # since it may be missing values to begin with.
        return (torch.rand_like(mask) > torch.rand_like(mask[:, :1])).float() * mask

    def calc_val_dict(self, val_loader, metric_f):
        val_metric = 0
        for x, y, m_data in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            m_data = m_data.to(self.device)
            val_metric += metric_f(self.predict(x, m_data), y) / len(val_loader)
        return val_metric, {"Predictive Metric": val_metric}

    def fit(self, train_data, val_data, ckpt_path, metric_f):
        # Main part of training.
        self.fit_parameters(train_data, val_data, ckpt_path, metric_f)
        # End the training, same for all models, finish training by running
        # zero acquisition on the val set and saving the value.
        self.load_state_dict(torch.load(osp.join(ckpt_path, "best_model.pt")))
        self.eval()
        if self.use_fixed_order:
            train_loader = DataLoader(
                train_data, batch_size=len(train_data), shuffle=False
            )
            self.find_fixed_order(train_loader, metric_f)
            torch.save(
                self.fixed_order_scores, osp.join(ckpt_path, "fixed_order_scores.pt")
            )
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        val_auc = self.run_zero_acquisition(val_loader, metric_f)
        torch.save(val_auc, osp.join(ckpt_path, "val_auc.pt"))
        print(f"\nTraining complete, Zero Acquisition AUC: {val_auc:.3f}")

    def fit_parameters(self, train_data, val_data, ckpt_path, metric_f):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_factor,
            cooldown=cooldown,
            min_lr=min_lr,
            patience=self.patience,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=self.batchsize,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=len(val_data),
            shuffle=False,
        )

        # Try to load in a checkpoint.
        try:
            ckpt = torch.load(osp.join(ckpt_path, "ckpt.pt"))
            start_epoch = ckpt["epoch"] + 1
            self.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"Restarting training from epoch {start_epoch}")
        except FileNotFoundError:
            start_epoch = 1
            print("Starting training from scratch")

        # Resume or start training.
        for epoch in range(start_epoch, self.epochs + 1):
            self.train()
            self.epoch_count = epoch - 1  # Can be used for beta annealing in VAEs.
            epoch_loss = 0
            for x, y, m_data in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                m_data = m_data.to(self.device)
                loss = self.loss_func(x, y, self.subsample_mask(m_data), m_data)
                loss.backward()
                epoch_loss += loss.item() / len(train_loader)
                optimizer.step()

            with torch.no_grad():
                self.eval()
                val_metric, val_dict = self.calc_val_dict(val_loader, metric_f)
            scheduler.step(val_metric)
            if val_metric == scheduler.best:
                torch.save(self.state_dict(), osp.join(ckpt_path, "best_model.pt"))

            # Save a checkpoint.
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, osp.join(ckpt_path, "ckpt.pt"))

            # Print results of this epoch.
            print(
                f"\nEpoch: {epoch}/{self.epochs}, Avg Loss: {epoch_loss:.3e}, ", end=""
            )
            print(f"Val Metric: {val_metric:.3f}|{scheduler.best:.3f}", end="")
            for key in val_dict:
                print(f", {key}: {val_dict[key]:.3f}", end="")

    def calculate_acquisition_scores(self, x, mask):
        # Default is the fixed ordering or random ordering.
        # Scores always need to be positive.
        if self.use_fixed_order:
            return self.fixed_order_scores.unsqueeze(0).repeat(mask.shape[0], 1)
        else:
            return torch.rand_like(mask)

    def acquire(self, x, mask_acq, mask_data, return_features=False):
        # We include mask_acq and mask_data, since mask_acq tells us what we have,
        # and mask_data tells us what out dataset lets us acquire.

        # Clear the cache before and after scoring since it can be a memory
        # intensive operation.
        self.clear_cache()
        feature_scores = self.calculate_acquisition_scores(x, mask_acq * mask_data)
        self.clear_cache()
        feature_scores += 1.0  # Add 1 to all scores to make sure we acquire something.
        feature_scores *= 1.0 - mask_acq  # Check if feature has been selected already.
        feature_scores *= (
            mask_data  # Based on our dataset, can we even acquire this feature?
        )

        # Anything that has not been acquired that we can will be at least 1.0. But
        # if we acquired everything that can be, we will have all zeros. We still
        # want to believe we have acquired features (even if they can't be), we
        # want to fill the mask. So we finally add a small value to anything not
        # acquired, and then torch argmax selects the first one.
        feature_scores += (1.0 - mask_acq) * 1e-6
        selected = torch.argmax(feature_scores, dim=-1)
        mask_acq = torch.max(mask_acq, F.one_hot(selected, self.num_features).float())
        if return_features:
            return mask_acq, selected
        else:
            return mask_acq

    def run_zero_acquisition(self, val_loader, metric_f):
        with torch.no_grad():
            val_auc = 0
            for x, y, m_data in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                m_data = m_data.to(self.device)
                m_acq = torch.zeros_like(m_data)
                val_metrics = [metric_f(self.predict(x, m_acq * m_data), y)]
                for _ in range(self.max_dim):
                    m_acq = self.acquire(x, m_acq, m_data)
                    val_metrics.append(metric_f(self.predict(x, m_acq * m_data), y))
                val_metrics = np.array(val_metrics)
                val_auc += sklearn_auc(np.arange(self.max_dim + 1), val_metrics) / (
                    len(val_loader) * self.max_dim
                )
        return val_auc

    def find_fixed_order(self, train_loader, metric_func):
        # Note this can be a slow process, it is best to do with as large a
        # batchsize as possible.
        print("\n\nFinding greedy fixed ordering")
        with torch.no_grad():
            possible_features = list(range(self.num_features))
            fixed_order = []
            while len(possible_features) > 1:
                print(
                    f"Remaining features: {len(possible_features)}/{self.num_features}"
                )
                best_metric = -1
                best_feature = -1
                for feature in possible_features:
                    avg_metric = 0.0
                    # Calculate the metric for each feature.
                    for x, y, m_data in train_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        m_data = m_data.to(self.device)
                        m_tmp = torch.zeros_like(m_data)
                        m_tmp[:, np.array(fixed_order + [feature]).astype(int)] = 1.0
                        m_tmp *= m_data
                        avg_metric += metric_func(self.predict(x, m_tmp), y) / len(
                            train_loader
                        )
                    if avg_metric > best_metric:
                        best_feature = int(feature)
                        best_metric = avg_metric
                # Add to the fixed order.
                fixed_order.append(best_feature)
                possible_features.remove(best_feature)
            # Add final feature.
            fixed_order.append(possible_features[0])
            # Convert the order to scores.
            scores = torch.zeros(self.num_features).to(self.device)
            for i in range(self.num_features):
                scores[fixed_order[i]] = self.num_features - i
            self.fixed_order_scores = scores.float()
