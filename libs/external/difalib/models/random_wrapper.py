import os
from copy import deepcopy

import torch
import torch.nn as nn

from .data_utils import load_data, prepare_data
from .model_utils import (
    assign_metric,
    assign_optimizer_scheduler,
    compute_batch_loss,
    compute_loss,
    compute_metric,
    finetune_step,
)
from .network_utils import get_imputation_model, get_pred_policy_network
from .utils import generate_random_feature_mask, mask_batch


class Model(object):
    def __init__(self, params, device):
        self.params = deepcopy(params)
        self.device = device
        # self.env = Env(self.params)
        self.prepare_data()
        self.pred_model, ac_network = get_pred_policy_network(
            self.params, self.data_parameters
        )
        self.pred_model = self.pred_model.to(self.device)
        assign_optimizer_scheduler(self)
        if self.params.use_imputation_model:
            self.imputation_model = self.load_imputation_model()
        assign_metric(self)

    def prepare_data(self):
        self.data_parameters = prepare_data(self)

    def load_imputation_model(self):
        if not os.path.exists(self.params.imputation_model):
            raise FileNotFoundError
        checkpoint = torch.load(self.params.imputation_model, map_location=self.device)
        model = get_imputation_model(
            checkpoint["params"], checkpoint["data_parameters"]
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def dataloaders(self, iters=None):
        return load_data(self)

    def train_step(self, batch):
        return self.step(batch, is_eval=False)

    def test_step(self, batch):
        with torch.no_grad():
            return self.step(batch, is_eval=True)

    def step(self, batch, is_eval):
        batch = batch["batch"]
        unobserved_mask = torch.ones_like(batch)
        not_done = torch.zeros(len(batch)).to(self.device) == 0.0  # B, all true
        acq_costs, ll_costs, n_features = (
            torch.zeros_like(not_done).float(),
            torch.zeros_like(not_done).float(),
            torch.zeros_like(not_done).float(),
        )
        actions, states = self.select_next_feature(batch, unobserved_mask)

        output = self.pred_model(states)
        loss = compute_batch_loss(self, output, batch)
        ll_costs += loss.detach()
        loss = loss.mean()
        if not is_eval:
            self.optimizer.zero_grad()
            loss.backward()
            if self.params.grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.pred_model.parameters(),
                    max_norm=self.params.grad_norm,
                    norm_type="inf",
                )
            self.optimizer.step()
            self.scheduler.step()
            # shared state, take step first, then the next one
        metric = compute_metric(self, output, batch)
        acq_costs, ll_costs = float(acq_costs.mean().detach().cpu()), float(
            ll_costs.mean().detach().cpu()
        )
        # Compute Feature State
        mean_feature_count = (
            torch.mean((1.0 - unobserved_mask[:, :-1].detach()) * 100.0, dim=0)
            .cpu()
            .numpy()
            .tolist()
        )
        return {
            self.metric: metric,
            "mean_feature_count": mean_feature_count,
            "acq_cost": acq_costs,
            "ll_cost": ll_costs,
            "reward": -acq_costs - ll_costs,
            "n_features": float(n_features.mean().detach().cpu()),
            "size": len(batch),
        }

    def finetune_step(self, batch, is_eval):
        return finetune_step(self, batch, is_eval)

    def select_next_feature(self, batch, unobserved_mask):
        nan_mask = torch.isnan(batch)
        input_mask = unobserved_mask.float() * (1.0 - nan_mask.float())
        input_mask = input_mask[:, :-1]
        actions = torch.multinomial(
            input_mask.float(), self.params.n_features, replacement=False
        )
        unobserved_mask.scatter_(dim=1, index=actions, value=0)
        state = self.compute_state(batch, unobserved_mask)
        assert not torch.any(
            nan_mask.gather(dim=1, index=actions)
        ), "nan  feature selected"
        # return mask done
        return actions, state

    def compute_imputation_params(self, batch, unobserved_mask):
        _, prior = self.imputation_model.make_latent_distributions(
            batch, unobserved_mask
        )
        latent = prior.sample()
        params = self.imputation_model.generative_network(latent)
        return params

    def compute_expected_future_state(self, batch, unobserved_mask):
        params_xo = self.compute_imputation_params(batch, unobserved_mask)
        # get feature distribution
        (
            xi_xo_entropy,
            xi_xo_mean,
            xi_xo_std,
        ) = self.imputation_model.rec_log_prob.get_entropy_mean_std(
            params_xo, entropy=True, mean=True, std=True, only_features=True
        )
        # compute future state
        future = []
        return future, {"mean": xi_xo_mean}

    def compute_state(self, batch, unobserved_mask):
        mbatch = mask_batch(batch, unobserved_mask)[:, :-1]
        mean = self.compute_expected_future_state(batch, unobserved_mask)[1]["mean"]
        mask = unobserved_mask[:, :-1].clone()
        mbatch = mbatch + mask * mean
        state = torch.cat([mbatch, mask], dim=-1)
        return state
