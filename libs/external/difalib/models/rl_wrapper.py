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
from .ppo import PPO, Memory
from .utils import generate_random_feature_mask, mask_batch


class Model(object):
    def __init__(self, params, device):
        self.params = deepcopy(params)
        self.device = device
        # self.env = Env(self.params)
        self.prepare_data()
        self.params.n_features = params.n_features = self.data_parameters["n_features"]
        self.pred_model, ac_network = get_pred_policy_network(
            self.params, self.data_parameters
        )
        self.pred_model = self.pred_model.to(self.device)
        assign_optimizer_scheduler(self)
        self.policy = PPO(ac_network.to(self.device), self.params)
        self.memory = Memory()
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

    def save(self):
        model_dir: str = os.path.join(self.params.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.policy.state_dict(),
                "model_state_dict": self.pred_model.state_dict(),
                "params": self.params,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "data_parameters": self.data_parameters,
            },
            os.path.join(model_dir, f"{self.params.name}.pt"),
        )

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
            with torch.no_grad():
                action, state, aux_info = self.select_next_feature(
                    batch, unobserved_mask, not_done
                )  # Shape (not_done,)
                done, acq_cost, ll_cost, reward, state = self.compute_reward(
                    batch[not_done], state, unobserved_mask[not_done], action, aux_info
                )
                self.memory.rewards.append(reward)
                self.memory.not_dones.append(not_done.clone())

            if states is None:
                states = torch.zeros(len(batch), len(state[0])).to(self.device)
            # for stop action cost is still calculated
            acq_costs[not_done] += acq_cost
            ll_costs[not_done] += ll_cost
            # not_done : (B,) , done: (B1, )
            # compute indices for not done and done
            # double slicing does not assign values
            not_done_indices = torch.where(not_done)[0]  # (B1,)
            new_done_indices = not_done_indices[done]
            new_not_done_indices = not_done_indices[~done]
            if self.params.cost_based:
                # for stop action, only not done samples are updated
                unobserved_mask[new_not_done_indices, action[~done]] = 0
                n_features[new_not_done_indices] += 1.0
            else:
                unobserved_mask[torch.arange(len(action)), action] = 0
                n_features += 1.0
            # for done samples, update state
            states[new_done_indices] = state[done]
            # finally update not_done
            not_done[new_done_indices] = False
        output = self.pred_model(states)
        loss = compute_loss(self, output, batch)
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
            self.policy.update(self.memory)
        metric = compute_metric(self, output, batch)
        self.memory.clear_memory()
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
            "output": (
                output.detach().to(device="cpu")
                if isinstance(output, torch.Tensor)
                else output
            ),
            "batch": (
                batch.detach().to(device="cpu")
                if isinstance(batch, torch.Tensor)
                else batch
            ),
        }

    def finetune_step(self, batch, is_eval):
        return finetune_step(self, batch, is_eval)

    def select_next_feature(self, batch, unobserved_mask, not_done):
        batch, unobserved_mask = batch[not_done], unobserved_mask[not_done]
        nan_mask = torch.isnan(batch)
        state, action_mask, aux_info = self.compute_state(batch, unobserved_mask)
        action = self.policy.policy_old.act(state, self.memory, action_mask)
        assert not torch.any(
            nan_mask[torch.arange(len(action)), action]
        ), "nan  feature selected"
        assert not torch.any(
            unobserved_mask[torch.arange(len(action)), action] == 0.0
        ), "already selected sample"

        # return mask done
        return action, state, aux_info

    def compute_reward(self, batch, state, unobserved_mask, action, aux_info):
        if self.params.cost_based:
            done = action == len(batch[0]) - 1
        else:
            done = (
                unobserved_mask.sum(dim=-1)
                == len(batch[0]) - self.params.n_features + 1
            )
        acq_cost, ll_cost, surr_reward = (
            torch.zeros(len(batch)).to(self.device),
            torch.zeros(len(batch)).to(self.device),
            torch.zeros(len(batch)).to(self.device),
        )
        if self.params.cost_based:
            # Acquisition cost
            acq_cost[~done] += self.params.acquisition_cost

        # Surrogate Reward
        if self.params.use_surrogate_reward:
            unobserved_mask_updated = unobserved_mask.clone()
            unobserved_mask_updated[~done, action[~done]] = 0.0
            params_xoi = self.compute_imputation_params(batch, unobserved_mask_updated)
            _, y_dist = self.imputation_model.rec_log_prob.label_sampler(
                params_xoi, sample_most_probable=False
            )
            y_xoi_entropy = y_dist.entropy().squeeze()
            surr_reward[~done] += (
                aux_info["y_xo_entropy"][~done]
                - self.params.gamma * y_xoi_entropy[~done]
            )
        # negative loglikelihood cost
        if torch.any(done):
            if not self.params.cost_based:
                new_unobserved_mask = unobserved_mask.clone()
                new_unobserved_mask[torch.arange(len(action)), action] = 0.0
                state, _, _ = self.compute_state(batch, new_unobserved_mask)
            output = self.pred_model(state[done])
            loss = compute_batch_loss(self, output, batch[done])
            ll_cost[done] += loss
        reward = -acq_cost - ll_cost + surr_reward
        return done, acq_cost, ll_cost, reward, state

    def compute_imputation_params(self, batch, unobserved_mask):
        _, prior = self.imputation_model.make_latent_distributions(
            batch, unobserved_mask
        )
        latent = prior.sample()
        params = self.imputation_model.generative_network(latent)
        return params

    def compute_expected_future_state(self, batch, unobserved_mask):
        params_xo = self.compute_imputation_params(batch, unobserved_mask)
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
        # get feature distribution
        (
            xi_xo_entropy,
            xi_xo_mean,
            xi_xo_std,
        ) = self.imputation_model.rec_log_prob.get_entropy_mean_std(
            params_xo, entropy=True, mean=True, std=True, only_features=True
        )
        xi_xoy_entropy, _, _ = self.imputation_model.rec_log_prob.get_entropy_mean_std(
            params_xoy, entropy=True, mean=False, std=False, only_features=True
        )
        # compute uis
        uis = xi_xo_entropy - xi_xoy_entropy  # B,D
        # mask observed ones
        observed_mask = unobserved_mask[:, :-1] == 0.0
        uis[observed_mask] = 0
        xi_xo_mean[observed_mask] = batch[:, :-1][observed_mask]
        xi_xo_std[observed_mask] = 0
        # compute future state
        future = torch.cat([uis, xi_xo_mean, xi_xo_std, y_probs], dim=-1)
        return future, {"y_xo_entropy": y_xo_entropy.squeeze()}

    def compute_state(self, batch, unobserved_mask):
        action_mask = unobserved_mask.clone()
        action_mask[torch.isnan(batch)] = 0.0
        if self.params.cost_based:
            action_mask[:, -1] = 1.0  # last dimension is stop action
        else:
            action_mask[:, -1] = 0.0
        mbatch = mask_batch(batch, unobserved_mask)
        state, info = (
            torch.cat([mbatch[:, :-1], unobserved_mask[:, :-1].clone()], dim=-1),
            {},
        )
        if self.params.use_aux_state:
            with torch.no_grad():
                future, info = self.compute_expected_future_state(
                    batch, unobserved_mask
                )
                state = torch.cat([state, future], dim=-1)
        return state, action_mask, info
