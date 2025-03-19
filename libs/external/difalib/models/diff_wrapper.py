import os
from copy import deepcopy

import torch
import torch.nn as nn

from .data_utils import load_data, prepare_data
from .difa import DIFA
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
        self.prepare_data()
        # self.env = Env(self.params)
        self.pred_model, ac_network = get_pred_policy_network(
            self.params, self.data_parameters
        )
        self.pred_model = self.pred_model.to(self.device)
        assign_optimizer_scheduler(self)

        self.policy = DIFA(ac_network.to(self.device), self.params)
        if self.params.use_imputation_model:
            self.imputation_model = self.load_imputation_model()
        assign_metric(self)

    def save(self):
        torch.save(
            {
                "policy_state_dict": self.policy.policy.state_dict(),
                "model_state_dict": self.pred_model.state_dict(),
                "params": self.params,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "data_parameters": self.data_parameters,
            },
            "models/{}.pt".format(self.params.name),
        )

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
        states = None
        while any(not_done):
            diff_actions, action, futures = self.select_next_feature(
                batch, unobserved_mask, not_done
            )  # Shape (not_done,)
            if self.params.cost_based:
                done = diff_actions[:, -1].detach() == 1.0
            else:
                done = (
                    unobserved_mask.sum(dim=-1)
                    == len(unobserved_mask[0]) - self.params.n_features
                )
            # double slicing does not assign values
            not_done_indices = torch.where(not_done)[0]  # (B1,)
            new_done_indices = not_done_indices[done]
            new_not_done_indices = not_done_indices[~done]
            if self.params.cost_based or any(done):
                term_states = self.compute_term_state(
                    batch[not_done], unobserved_mask[not_done], futures
                )
                output = self.pred_model(term_states)
                termination_loss = compute_batch_loss(self, output, batch[not_done])
            # Make Diff
            nonterm_states = self.compute_nonterm_state(
                batch[not_done], unobserved_mask[not_done], futures, diff_actions
            )
            output = self.pred_model(nonterm_states)
            observation_loss = compute_batch_loss(self, output, batch[not_done])
            +(self.params.acquisition_cost if self.params.cost_based else 0.0)
            # combined loss
            loss = observation_loss * (1.0 - diff_actions[:, -1])
            if self.params.cost_based:
                loss += termination_loss * diff_actions[:, -1]
            loss = loss.mean()
            # update mask, not_done, states, costs
            not_done[new_done_indices] = False
            unobserved_mask[new_not_done_indices, action[~done]] = 0
            if self.params.cost_based:
                acq_costs[new_not_done_indices] += self.params.acquisition_cost
            n_features[new_not_done_indices] += 1
            if states is None:
                states = torch.zeros(len(batch), len(nonterm_states[0])).to(self.device)
            if any(done):
                states[new_done_indices] = term_states[done].detach()
            if not self.params.cost_based and any(done):
                break
            if not is_eval:
                self.policy.update(loss)
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

    def select_next_feature(self, batch, unobserved_mask, not_done):
        batch, unobserved_mask = batch[not_done], unobserved_mask[not_done]
        nan_mask = torch.isnan(batch)
        with torch.no_grad():
            futures = self.compute_expected_future_state(batch, unobserved_mask)
        action_mask = unobserved_mask.clone()
        action_mask[torch.isnan(batch)] = 0.0
        if self.params.cost_based:
            action_mask[:, -1] = 1.0  # last dimension is stop action
        else:
            action_mask[:, -1] = 0.0
        state = self.compute_state(batch, unobserved_mask, futures)
        diff_action, action, entropy = self.policy.policy(state, action_mask)  #
        assert not torch.any(
            nan_mask[torch.arange(len(action)), action]
        ), "nan  feature selected"
        assert not torch.any(
            unobserved_mask[torch.arange(len(action)), action] == 0.0
        ), "already selected sample"

        # return mask done
        return diff_action, action, futures

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

        observed_mask = unobserved_mask[:, :-1] == 0.0
        xi_xo_mean[observed_mask] = batch[:, :-1][observed_mask]
        future, y_xo_entropy = None, None
        if self.params.use_aux_state:
            # sample most probable yhat
            yhat, y_dist = self.imputation_model.rec_log_prob.label_sampler(
                params_xo, sample_most_probable=True
            )
            y_xo_entropy = y_dist.entropy()
            if isinstance(y_dist, torch.distributions.normal.Normal):
                y_probs = torch.cat([y_dist.mean, y_dist.variance**0.5], dim=-1)
            else:
                y_probs = y_dist.probs
            # make y observed to compute H(x_i|y,x_o)
            batch_y, unobserved_mask_y = batch.clone(), unobserved_mask.clone()
            unobserved_mask_y[:, -1] = 0
            batch_y[:, -1] = yhat
            params_xoy = self.compute_imputation_params(batch_y, unobserved_mask_y)
            (
                xi_xoy_entropy,
                _,
                _,
            ) = self.imputation_model.rec_log_prob.get_entropy_mean_std(
                params_xoy, entropy=True, mean=False, std=False, only_features=True
            )
            # compute uis
            uis = xi_xo_entropy - xi_xoy_entropy  # B,D
            # mask observed ones
            uis[observed_mask] = 0
            xi_xo_std[observed_mask] = 0
            # compute future state
            future = torch.cat([uis, xi_xo_mean, xi_xo_std, y_probs], dim=-1)
        return {
            "mean": xi_xo_mean,
            "future": future,
            "y_xo_entropy": y_xo_entropy.squeeze(),
        }

    def compute_nonterm_state(self, batch, unobserved_mask, futures, diff_actions):
        diff_unobserved_mask = unobserved_mask[:, :-1].clone() - diff_actions[:, :-1]
        mean_batch = futures["mean"]
        batch_not_nan = batch.clone()
        batch_not_nan[torch.isnan(batch)] = 0.0
        diff_batch = (
            batch_not_nan[:, :-1] * (1.0 - diff_unobserved_mask)
            + mean_batch * diff_unobserved_mask
        )
        state = torch.cat([diff_batch, diff_unobserved_mask], dim=-1)
        if self.params.use_aux_state:
            future = futures["future"]
            state = torch.cat([state, future], dim=-1)
        return state

    def compute_term_state(self, batch, unobserved_mask, futures):
        # Termination loss
        mean_batch = futures["mean"]
        term_unobserved_mask = unobserved_mask[:, :-1].clone()
        batch_not_nan = batch.clone()
        batch_not_nan[torch.isnan(batch)] = 0.0
        term_batch = (
            batch_not_nan[:, :-1] * (1.0 - term_unobserved_mask)
            + mean_batch * term_unobserved_mask
        )
        state = torch.cat([term_batch, term_unobserved_mask], dim=-1)
        if self.params.use_aux_state:
            future = futures["future"]
            state = torch.cat([state, future], dim=-1)
        return state

    def compute_state(self, batch, unobserved_mask, futures):
        mbatch = mask_batch(batch, unobserved_mask)
        mean_batch = futures["mean"]
        mbatch = mbatch[:, :-1] + mean_batch * unobserved_mask[:, :-1]
        state = torch.cat([mbatch, unobserved_mask[:, :-1].clone()], dim=-1)
        if self.params.use_aux_state:
            future = futures["future"]
            state = torch.cat([state, future], dim=-1)
        return state
