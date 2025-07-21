"""Implementation for our model."""

from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .base import BaseModel
from .constants import min_sig
from .shared_functions import kl_01_loss
from .standard_layers import MLP

our_model_name = "IBFA"


class BlockLinear(nn.Module):
    """Individual linear layer in the partitioned model structure.
    in_dims: int, the number of input features per block, must be the same for
                  all blocks
    out_dims: int, the number of output features per block, must be the same for
                   all blocks
    num_blocks: int, the number of blocks in the block matrix
    """

    def __init__(self, in_dims, out_dims, num_blocks):
        super().__init__()
        # Initialize the weights and biases accoring to Kaiming Uniform, as per
        # PyTorch Linear initialisation (this improves learning significantly). See:
        # https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
        init_bound = 1 / (in_dims**0.5)
        self.weight = nn.Parameter(
            2 * init_bound * torch.rand(num_blocks, out_dims, in_dims) - init_bound
        )
        self.bias = nn.Parameter(
            2 * init_bound * torch.rand(num_blocks * out_dims) - init_bound
        )
        self.in_dims = in_dims

    def forward(self, x):
        # The input must be shape batch, num_groups*in_dims, we reshape to be used
        # with einsum. We then push it through the block matrix and reshape back.
        batchsize = x.shape[0]
        x = x.view(batchsize, -1, self.in_dims)
        return (
            torch.einsum("boi,xbi->xbo", self.weight, x).view(batchsize, -1) + self.bias
        )


class ContinuousBlockStochasticMLP(nn.Module):
    """Block MLP where we output a mean and diagonal standard deviation.
    This assumes all features are continuous, so all are size 1.
    We use softplus to convert a real number to a positive number. Specifically
    here the layer takes in a continuous vector, and a mask.
    num_con_features: int, the number of continuous features, i.e. number of blocks
    hidden_dim: int, the number of hidden units per block
    out_dims: int, the number of output features per block
    num_hidden: int, the number of hidden layers
    """

    def __init__(self, num_con_features, hidden_dim, out_dims, num_hidden):
        super().__init__()
        # Set up the network. Each feature is concatenated with its mask, so each
        # block is size 2.
        if num_hidden == 0:
            hidden_dim = 2
            self.to_hidden = nn.Identity()
        else:
            self.to_hidden = [BlockLinear(2, hidden_dim, num_con_features)]
            self.to_hidden.append(nn.ReLU())
            self.to_hidden.append(nn.BatchNorm1d(hidden_dim * num_con_features))
            for _ in range(num_hidden - 1):
                self.to_hidden.append(
                    BlockLinear(hidden_dim, hidden_dim, num_con_features)
                )
                self.to_hidden.append(nn.ReLU())
                self.to_hidden.append(nn.BatchNorm1d(hidden_dim * num_con_features))
            self.to_hidden = nn.Sequential(*self.to_hidden)

        self.h_to_mu = nn.Sequential(
            BlockLinear(hidden_dim, out_dims, num_con_features),
            nn.BatchNorm1d(out_dims * num_con_features),
        )
        self.h_to_sig = nn.Sequential(
            BlockLinear(hidden_dim, out_dims, num_con_features),
            nn.BatchNorm1d(out_dims * num_con_features),
        )

    def forward(self, x, mask):
        batchsize = x.shape[0]
        x = x.unsqueeze(-1)
        mask = mask.unsqueeze(-1)
        x = torch.cat([x * mask, mask], dim=-1).view(batchsize, -1)
        hidden = self.to_hidden(x)
        mu = self.h_to_mu(hidden)
        sig = F.softplus(self.h_to_sig(hidden)) + min_sig
        return mu, sig


class CategoricalBlockStochasticMLP(nn.Module):
    """Block MLP where we can split the output into a mean and standard deviation.
    Here the inputs are all integers represented as floats. Since all features are
    independently encoded for categorical, we can simply gather the weights for
    each feature based on that integer, no encoder needed. We use softplus to
    convert real number to positive.
    """

    def __init__(self, num_cat_features, out_dims, most_categories):
        super().__init__()
        # Since we do not stack layers, we initialise these parameters from a
        # standard normal, rather than Kaiming Unifrom.
        # We include +1 for a missing mask value.
        self.mu_param = nn.Parameter(
            torch.randn((most_categories + 1, num_cat_features, out_dims))
        )
        self.presig_param = nn.Parameter(
            torch.randn((most_categories + 1, num_cat_features, out_dims))
        )
        self.out_dims = out_dims

    def forward(self, x, mask):
        batchsize = x.shape[0]
        x = ((x + 1) * mask).long().unsqueeze(-1).expand(-1, -1, self.out_dims)
        mu = torch.gather(self.mu_param, 0, x).view(batchsize, -1)
        presig = torch.gather(self.presig_param, 0, x).view(batchsize, -1)
        return mu, F.softplus(presig) + min_sig


class OurModel(BaseModel):
    """Our model.
    NOTE: We do not use the shared input layers that other models use.
    """

    def __init__(self, config):
        super().__init__(config)
        self.latent_dim = config[
            "latent_dim"
        ]  # This is per feature needed to initialise encoders.
        self.ib_beta = config["ib_beta"]
        self.num_samples_train = config["num_samples_train"]
        self.num_samples_predict = config["num_samples_predict"]
        self.num_samples_acquire = config["num_samples_acquire"]
        self.size_limit = 3e7

        self.predictor = MLP(
            in_dim=config["latent_dim"] * self.num_features,
            hidden_dim=config["hidden_dim_predictor"],
            out_dim=self.out_dim,
            num_hidden=config["num_hidden_predictor"],
        )

        self.con_encoder = ContinuousBlockStochasticMLP(
            num_con_features=config["num_con_features"],
            hidden_dim=config["hidden_dim_encoder"],
            out_dims=config["latent_dim"],
            num_hidden=config["num_hidden_encoder"],
        )

        if self.mixed_input:
            self.last_con_index = config["num_con_features"]
            self.cat_encoder = CategoricalBlockStochasticMLP(
                num_cat_features=config["num_cat_features"],
                out_dims=config["latent_dim"],
                most_categories=config["most_categories"],
            )

    def encode(self, x, mask):
        if self.mixed_input:
            x_con = x[:, : self.last_con_index]
            mask_con = mask[:, : self.last_con_index]
            z_mu_con, z_sig_con = self.con_encoder(x_con, mask_con)

            x_cat = x[:, self.last_con_index :]
            mask_cat = mask[:, self.last_con_index :]
            z_mu_cat, z_sig_cat = self.cat_encoder(x_cat, mask_cat)

            z_mu = torch.cat([z_mu_con, z_mu_cat], dim=-1)
            z_sig = torch.cat([z_sig_con, z_sig_cat], dim=-1)
            return z_mu, z_sig
        return self.con_encoder(x, mask)

    def log_likelihood(self, x, mask, num_samples=100):
        z_mu, z_sig = self.encode(x, mask)
        samples = Normal(z_mu, z_sig).rsample([num_samples])
        # Need to reshape here, rather than view, otherwise get error:
        #   "RuntimeError: view size is not compatible with input tensor's size and
        #    stride (at least one dimension spans across two contiguous subspaces).
        #    Use .reshape(...) instead."
        # This is used at different points for EDDI and VAE as well.
        samples = samples.reshape(-1, z_mu.shape[-1])
        log_preds = F.log_softmax(self.predictor(samples), dim=-1)
        log_preds = log_preds.view(num_samples, -1, self.out_dim)
        log_preds = torch.logsumexp(log_preds, dim=0) - np.log(num_samples)
        return log_preds, z_mu, z_sig

    def predict(self, x, mask):
        # Predict has to give the distribution, not logits.
        return torch.exp(self.log_likelihood(x, mask, self.num_samples_predict)[0])

    def calculate_acquisition_scores_subset(self, x, mask):
        with torch.no_grad():
            z_mu, z_sig = self.encode(x, mask)
            samples = Normal(z_mu, z_sig).sample([self.num_samples_acquire])
        with torch.enable_grad():
            samples.requires_grad_(True)
            # Samples need to keep same name to take gradient, so we view but
            # assign new variable.
            samples_reshaped = samples.view(-1, z_mu.shape[-1])
            preds = F.log_softmax(self.predictor(samples_reshaped), dim=-1)
            preds = preds.view(self.num_samples_acquire, -1, self.out_dim)
            preds = torch.logsumexp(preds, dim=0) - np.log(self.num_samples_acquire)
            preds = torch.exp(preds)
            preds_sum = torch.sum(preds, dim=0)
            # Calculate all of the proposed importances.
            # NOTE: we have a for loop over the labels here which is slower than
            # creating the full Jacobian with functorch.jacrev or
            # torch.autograd.functional.jacobian for example.
            # However this is currently weighed against the memory requirements
            # for storing a full Jacobian. This can be changed if required.
            scores = 0
            for c in range(self.out_dim):
                grads = torch.autograd.grad(
                    preds_sum[c], samples, retain_graph=(c != self.out_dim - 1)
                )[0]
                with torch.no_grad():
                    grads = grads.view(
                        self.num_samples_acquire, -1, self.num_features, self.latent_dim
                    )
                    grads = torch.sum(grads**2, dim=-1) ** 0.5
                    grads = grads / (torch.sum(grads, dim=-1, keepdim=True) + 1e-8)
                    grads = torch.mean(grads, dim=0)
                    scores += grads * preds[:, c : c + 1]
        return scores

    def calculate_acquisition_scores(self, x, mask):
        # We split the validation set if it is too large, since this
        # is an extremely memory intensive operation.
        with torch.no_grad():
            batchsize = x.shape[0]
            size_samples = (
                batchsize
                * self.num_samples_acquire
                * self.num_features
                * self.latent_dim
            )
            if size_samples <= self.size_limit:
                ids = [torch.arange(batchsize)]
            else:
                ids = np.array_split(
                    np.arange(batchsize), ceil(size_samples / self.size_limit)
                )
            scores = []
            for i in ids:
                # Clear cache before and after this, since it is memory intensive.
                self.clear_cache()
                scores.append(self.calculate_acquisition_scores_subset(x[i], mask[i]))
                self.clear_cache()
            scores = torch.cat(scores, dim=0)
            return scores

    def loss_func(self, x, y, mask, data_mask=None):
        log_likelihood, z_mu, z_sig = self.log_likelihood(
            x, mask, self.num_samples_train
        )
        loss1 = F.nll_loss(log_likelihood, y) + self.ib_beta * kl_01_loss(z_mu, z_sig)
        # return loss1  # If we want to train without looking at our acquisitions.
        # This encourages the model to do well on acquisitions it can expect to see.
        with torch.no_grad():
            mask = self.acquire(x, mask, data_mask)
        log_likelihood, z_mu, z_sig = self.log_likelihood(
            x, mask, self.num_samples_train
        )
        loss2 = F.nll_loss(log_likelihood, y) + self.ib_beta * kl_01_loss(z_mu, z_sig)
        return (loss1 + loss2) / 2

    def calc_val_dict(self, val_loader, metric_f):
        val_metric = 0
        ib = 0
        for x, y, mask in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)
            log_likelihood, z_mu, z_sig = self.log_likelihood(
                x, mask, self.num_samples_predict
            )
            val_metric += metric_f(torch.exp(log_likelihood), y) / len(val_loader)
            ib += kl_01_loss(z_mu, z_sig) / (
                len(val_loader) * self.latent_dim * self.num_features
            )
        val_auc = self.run_zero_acquisition(val_loader, metric_f)
        return val_auc, {"Predictive Metric": val_metric, "IB": ib, "Val Auc": val_auc}
