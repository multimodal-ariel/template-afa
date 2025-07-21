"""DIME from the ICLR 2024 paper: https://arxiv.org/abs/2306.03301
Code: https://github.com/suinleelab/DIME/tree/main

The idea is that we have a predictive model, and a value model. And like
in Mutual Neural Estimation the value model can be used to estimate the mutual
information directly.

Most code based on: https://github.com/suinleelab/DIME/blob/main/dime/cmi_estimator.py
A lot of code copied from our GDFS implementation.
"""

import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .base import BaseModel
from .constants import cooldown, log_eps, lr_factor, min_lr
from .standard_layers import MLP


class DIME(BaseModel):
    """DIME implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.pretrain_epochs = config["pretrain_epochs"]
        self.pretrain_lr = config["pretrain_lr"]
        self.eps_scale = config["eps_scale"]

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
            self.value_layers = nn.Sequential(
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
            self.value_layers = MLP(
                in_dim=self.in_dim,
                hidden_dim=config["hidden_dim"],
                out_dim=self.num_features,
                num_hidden=config["num_hidden"],
            )

    def predictor(self, x, mask):
        return self.predictor_layers(self.input_layer(x, mask))

    def predict(self, x, mask):
        return F.softmax(self.predictor(x, mask), dim=-1)

    def value(self, x, mask):
        # Note this is unscaled. But scaling is included in the calculate
        # acquisition scores and during training loop.
        return torch.sigmoid(self.value_layers(self.input_layer(x, mask)))

    def calculate_acquisition_scores(self, x, mask):
        with torch.no_grad():
            H = self.predict(x, mask)
            H = -torch.sum(H * torch.log(H + log_eps), dim=-1).unsqueeze(-1)
            return H * self.value(x, mask)

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
            start_eps_id = ckpt["eps_id"]
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
                    start_eps_id = 0
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
                    "eps_id": 0,
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

        # Main training of value prediction.
        eps_progression = self.eps_scale * np.array([1.0, 0.25, 0.05, 0.005])
        num_eps_steps = len(eps_progression)

        print(
            f"\nStarting main training from eps {eps_progression[start_eps_id]:.3e} ",
            end="",
        )
        print(f"({start_eps_id+1}/{num_eps_steps}), ", end="")
        if start_main_epoch == self.epochs + 1:
            print(f"resetting for next epsilon")
        else:
            print(f"epoch {start_main_epoch}/{self.epochs}\n")

        for eps_id in range(start_eps_id, num_eps_steps):
            eps = eps_progression[eps_id]
            for epoch in range(start_main_epoch, self.epochs + 1):
                self.train()
                epoch_pred_loss = 0
                epoch_value_loss = 0
                for x, y, m_data in train_loader:
                    optimizer.zero_grad()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    m_data = m_data.to(self.device)
                    m_acq = torch.zeros_like(m_data)

                    p_curr = self.predictor(x, m_acq * m_data)
                    loss_prev = F.cross_entropy(p_curr, y, reduction="none")
                    loss_tmp = torch.mean(loss_prev) / (self.max_dim + 1)
                    loss_tmp.backward()
                    epoch_pred_loss += loss_tmp.item() / len(train_loader)
                    loss_prev = loss_prev.detach()

                    for _ in range(self.max_dim):
                        p_curr = F.softmax(p_curr, dim=-1).detach()
                        H_curr = -torch.sum(
                            p_curr * torch.log(p_curr + log_eps), dim=-1
                        ).unsqueeze(-1)
                        cmi = H_curr.detach() * self.value(x, m_acq * m_data)
                        cmi = cmi * (1 - m_acq) * m_data + (1 - m_acq) * 1e-6
                        cmi_max = F.one_hot(
                            torch.argmax(cmi, dim=-1), num_classes=self.num_features
                        ).float()

                        unif = (
                            torch.rand_like(cmi) * (1 - m_acq) * m_data
                            + (1 - m_acq) * 1e-6
                        )
                        unif = F.one_hot(
                            torch.argmax(unif, dim=-1), num_classes=self.num_features
                        ).float()

                        c_or_u = torch.bernoulli(torch.full_like(cmi[:, 0:1], 1 - eps))
                        m_update = c_or_u * cmi_max + (1 - c_or_u) * unif
                        m_acq = torch.max(m_acq, m_update)

                        p_curr = self.predictor(x, m_acq * m_data)
                        loss_curr = F.cross_entropy(p_curr, y, reduction="none")
                        loss_tmp = torch.mean(loss_curr) / (self.max_dim + 1)
                        loss_tmp.backward()
                        epoch_pred_loss += loss_tmp.item() / len(train_loader)
                        loss_curr = loss_curr.detach()

                        delta = (loss_prev - loss_curr).detach()
                        cmi_preds = torch.sum(cmi * m_update, dim=-1)
                        cmi_loss_tmp = (
                            torch.mean((cmi_preds - delta) ** 2) / self.max_dim
                        )
                        cmi_loss_tmp.backward()
                        epoch_value_loss += cmi_loss_tmp.item() / len(train_loader)
                        loss_prev = loss_curr

                    optimizer.step()

                self.eval()
                val_auc = self.run_zero_acquisition(val_loader, metric_f)
                scheduler.step(val_auc)

                if val_auc > best_hard_val_auc:
                    best_hard_val_auc = val_auc
                    torch.save(self.state_dict(), osp.join(ckpt_path, "best_model.pt"))

                # Save a checkpoint.
                ckpt = {
                    "pretraining_epoch": self.pretrain_epochs,
                    "eps_id": eps_id,
                    "main_epoch": epoch,
                    "best_hard_val_auc": best_hard_val_auc,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(ckpt, osp.join(ckpt_path, "ckpt.pt"))

                print(
                    f"Eps: {eps:.3e} ({eps_id+1}/{num_eps_steps}), Epoch: {epoch}/{self.epochs}, ",
                    end="",
                )
                print(
                    f"Val AUC: {val_auc:.3f}|{scheduler.best:.3f}|{best_hard_val_auc:.3f}, ",
                    end="",
                )
                print(
                    f"Pred Loss: {epoch_pred_loss:.3f}, CMI Loss: {epoch_value_loss:.3e}"
                )

            # Get ready for the next eps_id
            print("")
            start_main_epoch = 1
            optimizer = Adam(self.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=lr_factor,
                cooldown=cooldown,
                min_lr=min_lr,
                patience=self.patience,
            )
