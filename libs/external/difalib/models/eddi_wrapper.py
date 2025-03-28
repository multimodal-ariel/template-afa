import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import kl_divergence

from .data_utils import load_data, prepare_data
from .model_utils import (
    assign_metric,
    assign_optimizer_scheduler,
    compute_batch_loss,
    compute_loss,
    compute_metric,
    finetune_step,
)
from .network_utils import get_imputation_model, get_pred_network
from .utils import generate_random_feature_mask, mask_batch


class Model(object):
    def __init__(self, params, device):
        self.params = deepcopy(params)
        self.device = device
        self.model = self.load_imputation_model()
        self.prepare_data()
        self.model.eval()
        self.pred_model = get_pred_network(self.params, self.data_parameters).to(
            self.device
        )
        assign_optimizer_scheduler(self)
        assign_metric(self)

    def prepare_data(self):
        self.data_parameters = prepare_data(self)

    def load_imputation_model(self):
        checkpoint = torch.load(self.params.imputation_model, map_location=self.device)
        model = get_imputation_model(
            checkpoint["params"], checkpoint["data_parameters"]
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def finetune_step(self, batch, is_eval):
        return finetune_step(self, batch, is_eval)

    def compute_state(self, batch, unobserved_mask):
        mbatch = mask_batch(batch, unobserved_mask)
        state = torch.cat([mbatch[:, :-1], unobserved_mask[:, :-1].clone()], dim=-1)
        return state

    def dataloaders(self, iters=None):
        return load_data(self)

    def train_step(self, batch):
        return {"size": 1, "ll": 1, self.metric: 0.0}
        # return self.step(batch, is_eval=False)

    def test_step(self, batch):
        start_time = time.time()
        with torch.no_grad():
            out = self.step(batch, is_eval=True)
        end_time = time.time()
        print("Time Taken: {} hours".format((end_time - start_time) / 3600.0))
        return out

    def step(self, batch, is_eval):
        batch = batch["batch"]
        unobserved_mask = torch.ones_like(batch)
        metric_for_each_step = []
        ll_for_each_step = []
        outs_l = list()
        for action_step in range(self.params.n_features):
            metric, action = self.select_next_feature(batch, unobserved_mask)
            unobserved_mask[torch.arange(len(batch)), action] = 0
            state = self.compute_state(batch, unobserved_mask)
            if is_eval or action_step < self.params.n_features - 1:
                with torch.no_grad():
                    output = self.pred_model(state)
            else:
                output = self.pred_model(state)
            outs_l.append(
                output.detach().to(device="cpu")
                if isinstance(output, torch.Tensor)
                else output
            )
            loss = compute_loss(self, output, batch)
            metric = compute_metric(self, output, batch)
            metric_for_each_step.append(metric)
            ll_for_each_step.append(float(-loss.detach().cpu()))
        if not is_eval:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        output = {
            self.metric: metric_for_each_step[-1],
            "size": len(batch),
            "ll": ll_for_each_step,
            "outs_l": outs_l,
        }
        for idx in range(len(metric_for_each_step)):
            output[self.metric + "_" + str(idx + 1)] = metric_for_each_step[idx]

        return output

    def select_next_feature(self, batch, unobserved_mask):
        nan_mask = torch.isnan(batch)
        with torch.no_grad():
            batch_uis = torch.zeros_like(batch)[:, :-1]
            _, prior_xo = self.model.make_latent_distributions(
                mask_batch(batch, unobserved_mask), unobserved_mask
            )
            latent = prior_xo.rsample()
            params_xo = self.model.generative_network(latent)
            sample_xo = self.model.rec_log_prob.sampler(
                params_xo, sample_most_probable=True
            )
            mask_y, batch_y = unobserved_mask.clone(), batch.clone()
            mask_y[:, -1] = 0
            batch_y[:, -1] = sample_xo[:, -1]
            _, prior_xoy = self.model.make_latent_distributions(
                mask_batch(batch_y, mask_y), mask_y, no_proposal=True
            )

            for col in range(len(batch[0]) - 1):
                self._get_acquisition_metric(
                    batch_uis,
                    col,
                    unobserved_mask,
                    sample_xo,
                    batch,
                    prior_xo,
                    prior_xoy,
                )
            batch_uis[unobserved_mask[:, :-1] == 0] = -1e32
            batch_uis[nan_mask[:, :-1]] = -1e32
            if self.data_parameters["n_classes"] > 1:
                metric = torch.sum(
                    sample_xo[:, -1].long() == batch[:, -1].long()
                ) / float(len(sample_xo))
            else:  # "-mse"
                metric = -torch.sum((sample_xo[:, -1] - batch[:, -1]) ** 2.0) / len(
                    batch_uis
                )
            action = torch.argmax(batch_uis, dim=-1)
            assert not torch.any(
                nan_mask[torch.arange(len(action)), action]
            ), "nan  feature selected"
            assert not torch.any(
                unobserved_mask[torch.arange(len(action)), action] == 0.0
            ), "already selected sample"
            return float(metric.detach().cpu()), action

    def _get_acquisition_metric(
        self, batch_uis, col, unobserved_mask, sample_xo, batch, prior_xo, prior_xoy
    ):
        col_mask = unobserved_mask.clone()
        col_mask[:, col] = 0
        col_batch = batch.clone()
        col_batch[:, col] = sample_xo[:, col]
        _, prior_xoi = self.model.make_latent_distributions(
            mask_batch(col_batch, col_mask), col_mask, no_proposal=True
        )
        batch_uis[:, col] = (
            kl_divergence(prior_xoi, prior_xo).view(batch.shape[0], -1).sum(-1)
        )
        # p(z|x_o,x_i,y)|| p(z|x_o,y)
        col_batch[:, -1] = sample_xo[:, -1]
        col_mask[:, -1] = 0
        _, prior_xoiy = self.model.make_latent_distributions(
            mask_batch(col_batch, col_mask), col_mask, no_proposal=True
        )
        batch_uis[:, col] -= (
            kl_divergence(prior_xoiy, prior_xoy).view(batch.shape[0], -1).sum(-1)
        )
        return
