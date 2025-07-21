"""GDFS model implementation to be used in our codebase.
Paper: https://arxiv.org/abs/2301.00557
Code: https://github.com/iancovert/dynamic-selection/tree/main

The majority of this code is adapted specifically from:
https://github.com/iancovert/dynamic-selection/blob/main/dynamic_selection/greedy.py
"""

import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc as sklearn_auc
from torch.distributions import RelaxedOneHotCategorical
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .base import BaseModel
from .constants import cooldown, lr_factor, min_lr
from .standard_layers import MLP


class GDFS(BaseModel):
    """GDFS model."""

    def __init__(self, config):
        super().__init__(config)
        self.pretrain_epochs = config["pretrain_epochs"]
        self.pretrain_lr = config["pretrain_lr"]
        self.temp_scale = config["temp_scale"]

        if config["share_parameters"]:
            to_hidden = nn.Sequential(
                MLP(
                    in_dim=self.in_dim,
                    hidden_dim=config["hidden_dim"],
                    out_dim=config["hidden_dim"],
                    num_hidden=int(config["num_hidden"] / 2),
                ),
                nn.ReLU(),
                nn.BatchNorm1d(config["hidden_dim"]),
            )
            self.predictor_layers = nn.Sequential(
                to_hidden,
                MLP(
                    in_dim=config["hidden_dim"],
                    hidden_dim=config["hidden_dim"],
                    out_dim=self.out_dim,
                    num_hidden=config["num_hidden"] - int(config["num_hidden"] / 2),
                ),
            )
            self.selector_layers = nn.Sequential(
                to_hidden,
                MLP(
                    in_dim=config["hidden_dim"],
                    hidden_dim=config["hidden_dim"],
                    out_dim=self.num_features,
                    num_hidden=config["num_hidden"] - int(config["num_hidden"] / 2),
                ),
            )
        # If we are not sharing parameters we have two separate MLPs.
        else:
            self.predictor_layers = MLP(
                in_dim=self.in_dim,
                hidden_dim=config["hidden_dim"],
                out_dim=self.out_dim,
                num_hidden=config["num_hidden"],
            )
            self.selector_layers = MLP(
                in_dim=self.in_dim,
                hidden_dim=config["hidden_dim"],
                out_dim=self.num_features,
                num_hidden=config["num_hidden"],
            )

    def predictor(self, x, mask):
        return self.predictor_layers(self.input_layer(x, mask))

    def selector(self, x, mask):
        return self.selector_layers(self.input_layer(x, mask))

    def predict(self, x, mask):
        return F.softmax(self.predictor(x, mask), dim=-1)

    def calculate_acquisition_scores(self, x, mask):
        return F.softmax(self.selector(x, mask), dim=-1)

    def fit_parameters(self, train_data, val_data, ckpt_path, metric_f):
        optimizer = Adam(self.parameters(), lr=self.pretrain_lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_factor,
            cooldown=cooldown,
            min_lr=min_lr,
            patience=3 * self.patience,
        )
        train_loader = DataLoader(train_data, batch_size=self.batchsize, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

        # Try to load in a checkpoint.
        try:
            ckpt = torch.load(osp.join(ckpt_path, "ckpt.pt"))
            pretraining_start_epoch = ckpt["pretraining_epoch"] + 1
            pretraining_complete = pretraining_start_epoch > self.pretrain_epochs
            start_temp_id = ckpt["temp_id"]
            start_main_epoch = ckpt["main_epoch"] + 1
            best_hard_val_auc = ckpt["best_hard_val_auc"]
            self.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except FileNotFoundError:
            pretraining_start_epoch = 1
            pretraining_complete = False

        # Pretraining the predictor.
        if not pretraining_complete:
            print(
                f"Starting Pretraining from epoch {pretraining_start_epoch}/{self.pretrain_epochs}\n"
            )
            for epoch in range(pretraining_start_epoch, self.pretrain_epochs + 1):
                self.train()
                epoch_loss = 0
                for x, y, m_data in train_loader:
                    optimizer.zero_grad()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    m_data = m_data.to(self.device)
                    loss = F.cross_entropy(
                        self.predictor(x, self.subsample_mask(m_data)), y
                    )
                    loss.backward()
                    epoch_loss += loss.item() / len(train_loader)
                    optimizer.step()

                with torch.no_grad():
                    self.eval()
                    val_metric = 0
                    for x, y, m_data in val_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        m_data = m_data.to(self.device)
                        val_metric += metric_f(self.predict(x, m_data), y) / len(
                            val_loader
                        )

                scheduler.step(val_metric)
                best_val_metric = scheduler.best
                if val_metric == best_val_metric:
                    torch.save(self.state_dict(), osp.join(ckpt_path, "best_model.pt"))

                # Save a checkpoint.
                # After pretraining we want to start the main training from the best model,
                # with a new optimizer and scheduler. We set those here, so if training
                # fails between now and the first full main epoch, the checkpoint stores
                # the reset optimizer/scheduler and best current model.
                if epoch == self.pretrain_epochs:
                    start_temp_id = 0
                    start_main_epoch = 1
                    best_hard_val_auc = 0
                    self.load_state_dict(
                        torch.load(osp.join(ckpt_path, "best_model.pt"))
                    )
                    optimizer = Adam(self.parameters(), lr=self.lr)
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=lr_factor,
                        cooldown=cooldown,
                        min_lr=min_lr,
                        patience=self.patience,
                    )
                ckpt = {
                    "pretraining_epoch": epoch,
                    "temp_id": 0,
                    "main_epoch": 0,
                    "best_hard_val_auc": 0,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(ckpt, osp.join(ckpt_path, "ckpt.pt"))

                # Print results of this epoch.
                print(f"Pretraining Epoch: {epoch}/{self.pretrain_epochs}, ", end="")
                print(f"Avg Loss: {epoch_loss:.3f}, ", end="")
                print(f"Val Metric: {val_metric:.3f}|{best_val_metric:.3f}")

        # Main training of the selection process.
        num_temp_steps = 5
        temp_progression = self.temp_scale * np.geomspace(1.0, 0.1, num_temp_steps)

        print(
            f"\nStarting main training from temperature {temp_progression[start_temp_id]:.3e} ",
            end="",
        )
        print(f"({start_temp_id+1}/{num_temp_steps}), ", end="")
        if start_main_epoch == self.epochs + 1:
            print(f"resetting for next temperature")
        else:
            print(f"epoch {start_main_epoch}/{self.epochs}\n")

        for temp_id in range(start_temp_id, num_temp_steps):
            temp = temp_progression[temp_id]
            for epoch in range(start_main_epoch, self.epochs + 1):
                self.train()
                epoch_loss = 0
                for x, y, m_data in train_loader:
                    optimizer.zero_grad()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    m_data = m_data.to(self.device)
                    m_acq = torch.zeros_like(m_data)

                    for _ in range(self.max_dim):
                        select_logits = self.selector(x, m_acq * m_data)
                        m_av_sel = m_data * (
                            1 - m_acq
                        )  # Mask for available and not selected features.
                        select_logits = select_logits * m_av_sel - 1e6 * (1 - m_av_sel)
                        m_soft = torch.max(
                            m_acq,
                            RelaxedOneHotCategorical(
                                temp, logits=select_logits
                            ).rsample(),
                        )

                        loss = (
                            F.cross_entropy(self.predictor(x, m_soft * m_data), y)
                            / self.max_dim
                        )
                        loss.backward()
                        epoch_loss += loss.item() / len(train_loader)

                        # If nothing is available that we haven't already collected we want
                        # to choose an arbitrary not selected feature.
                        select_logits += (1 - m_acq) * 1e5
                        m_acq = torch.max(
                            m_acq,
                            F.one_hot(
                                torch.argmax(select_logits, dim=-1),
                                num_classes=self.num_features,
                            ).float(),
                        )

                    optimizer.step()

                with torch.no_grad():
                    self.eval()
                    soft_val_metric = 0
                    hard_val_auc = 0
                    for x, y, m_data in val_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        m_data = m_data.to(self.device)
                        m_acq = torch.zeros_like(m_data)
                        soft_val_list = [metric_f(self.predict(x, m_acq * m_data), y)]
                        hard_val_list = [soft_val_list[0]]
                        for _ in range(self.max_dim):
                            select_logits = self.selector(x, m_acq * m_data)
                            m_av_sel = (1 - m_acq) * m_data
                            select_logits = select_logits * m_av_sel - 1e6 * (
                                1 - m_av_sel
                            )
                            m_soft = torch.max(
                                m_acq,
                                RelaxedOneHotCategorical(
                                    temp, logits=select_logits
                                ).sample(),
                            )
                            soft_val_list.append(
                                metric_f(self.predict(x, m_soft * m_data), y)
                            )

                            select_logits += (1 - m_acq) * m_data * 1e5
                            m_acq = torch.max(
                                m_acq,
                                F.one_hot(
                                    torch.argmax(select_logits, dim=-1),
                                    num_classes=self.num_features,
                                ).float(),
                            )
                            hard_val_list.append(
                                metric_f(self.predict(x, m_acq * m_data), y)
                            )

                        soft_val_list = np.array(soft_val_list)
                        hard_val_list = np.array(hard_val_list)

                        soft_val_metric += sklearn_auc(
                            np.arange(self.max_dim + 1), soft_val_list
                        ) / (len(val_loader) * self.max_dim)
                        hard_val_auc += sklearn_auc(
                            np.arange(self.max_dim + 1), hard_val_list
                        ) / (len(val_loader) * self.max_dim)

                scheduler.step(soft_val_metric)
                if soft_val_metric == scheduler.best:
                    torch.save(
                        self.state_dict(), osp.join(ckpt_path, "best_soft_model.pt")
                    )

                if hard_val_auc > best_hard_val_auc:
                    best_hard_val_auc = hard_val_auc
                    torch.save(self.state_dict(), osp.join(ckpt_path, "best_model.pt"))

                # Save a checkpoint.
                ckpt = {
                    "pretraining_epoch": self.pretrain_epochs,
                    "temp_id": temp_id,
                    "main_epoch": epoch,
                    "best_hard_val_auc": best_hard_val_auc,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(ckpt, osp.join(ckpt_path, "ckpt.pt"))

                # Print info about this epoch.
                print(
                    f"Temp: {temp:.3e} ({temp_id+1}/{num_temp_steps}), Epoch: {epoch}/{self.epochs}, ",
                    end="",
                )
                print(f"Avg Loss: {epoch_loss:.3f}, ", end="")
                print(
                    f"Soft Val Metric: {soft_val_metric:.3f}|{scheduler.best:.3f}, ",
                    end="",
                )
                print(f"Hard Val AUC: {hard_val_auc:.3f}|{best_hard_val_auc:.3f}")

            # Get ready for the next temp_id
            print("")
            start_main_epoch = 1
            self.load_state_dict(torch.load(osp.join(ckpt_path, "best_soft_model.pt")))
            optimizer = Adam(self.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=lr_factor,
                cooldown=cooldown,
                min_lr=min_lr,
                patience=self.patience,
            )
