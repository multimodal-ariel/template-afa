"""Implementation of EDDI model.
Paper: https://arxiv.org/abs/1809.11142
Original code: https://github.com/Microsoft/EDDI
This is a large generative model that estimates the mutual information between
the label and the input features, by taking the expected KL divergence
in the latent space.
"""

from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .shared_functions import kl_01_loss, kl_div, nll_con_f, nll_mixed_f, sample_mixed
from .standard_layers import MLP, StochasticMLP


class ContinuousXtoC(nn.Module):
    """The part of EDDI that takes X and mask and gives aggregated C.
    NOTE: This assumes we have continuous features.
    """

    def __init__(self, num_con_features, hidden_dim, c_dim, num_hidden):
        super().__init__()
        self.c_dim = c_dim
        self.num_con_features = num_con_features
        self.E = nn.Parameter(torch.randn(1, num_con_features, c_dim))
        self.s_to_c = MLP(
            in_dim=c_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=c_dim,
            num_hidden=num_hidden,
        )

    def forward(self, x, mask):
        batchsize = x.shape[0]
        x = x.unsqueeze(-1)
        x = torch.cat([x, x * self.E], dim=-1).view(
            batchsize * self.num_con_features, -1
        )
        x = self.s_to_c(x).view(batchsize, self.num_con_features, -1)
        return torch.sum(x * mask.unsqueeze(-1), dim=1)


class CategoricalXtoC(nn.Module):
    """The part of EDDI that takes categorical X and mask and gives aggregated C.
    NOTE: This assumes categorical only features. There's no real need for this
    to go through a deep encoder, we can just gather the indices like with our
    model.
    """

    def __init__(self, num_cat_features, c_dim, most_categories):
        super().__init__()
        self.E = nn.Parameter(torch.randn(most_categories, num_cat_features, c_dim))
        self.c_dim = c_dim
        self.num_cat_features = num_cat_features

    def forward(self, x, mask):
        x = x.long().unsqueeze(-1).expand(-1, -1, self.c_dim)
        x = torch.gather(self.E, 0, x)
        return torch.sum(x * mask.unsqueeze(-1), dim=1)


class EDDI(BaseModel):
    """EDDI base model. Uses generative model to estimate mutual information.
    This is used to score the features and acquire them.
    NOTE: We use one sample from z_dist to train, as is done in EDDI.
    """

    def __init__(self, config):
        super().__init__(config)
        self.latent_dim = config["latent_dim"]
        self.obs_sig = config["sig"]
        self.num_acquisition_samples = config["num_acquisition_samples"]
        # NOTE this has been previously set to 1e7 to improve speed.
        self.size_limit = (
            1e6  # Max size of tensor to avoid memory issues, when calculating scores.
        )

        if self.mixed_input:
            self.last_con_index = config["num_con_features"]
            self.most_categories = config["most_categories"]
            self.cat_x_to_c = CategoricalXtoC(
                num_cat_features=config["num_cat_features"],
                c_dim=config["c_dim"],
                most_categories=config["most_categories"],
            )

        # Encoder parts of the model.
        self.con_x_to_c = ContinuousXtoC(
            num_con_features=config["num_con_features"],
            hidden_dim=config["hidden_dim_encoder"],
            c_dim=config["c_dim"],
            num_hidden=int(config["num_hidden_encoder"] / 2),
        )
        self.c_to_zdist = StochasticMLP(
            in_dim=config["c_dim"],
            hidden_dim=config["hidden_dim_encoder"],
            out_dim=config["latent_dim"],
            # Share the hidden layers with the s to c part of the encoder.
            num_hidden=config["num_hidden_encoder"]
            - int(config["num_hidden_encoder"] / 2),
        )
        self.E_y = nn.Parameter(torch.randn(self.out_dim, config["c_dim"]))

        # Decoder parts of the model.
        self.z_to_h = nn.Sequential(
            MLP(
                in_dim=config["latent_dim"],
                hidden_dim=config["hidden_dim_decoder"],
                out_dim=config["hidden_dim_decoder"],
                num_hidden=config["num_hidden_decoder"] - 1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(config["hidden_dim_decoder"]),
        )
        self.h_to_y = nn.Linear(config["hidden_dim_decoder"], self.out_dim)
        self.h_to_x = nn.Linear(
            config["hidden_dim_decoder"],
            config["num_con_features"]
            + config["num_cat_features"] * config["most_categories"],
        )

    def nll_f(self, x, recon, mask):
        if self.mixed_input:
            return nll_mixed_f(x, recon, mask, self.obs_sig, self.last_con_index)
        return nll_con_f(x, recon, mask, self.obs_sig)

    def sample_from_recon(self, recon):
        if self.mixed_input:
            return sample_mixed(recon, self.last_con_index, self.most_categories)
        return recon

    def x_to_c(self, x, mask):
        if self.mixed_input:
            x_con = x[:, : self.last_con_index]
            mask_con = mask[:, : self.last_con_index]

            x_cat = x[:, self.last_con_index :]
            mask_cat = mask[:, self.last_con_index :]
            return self.con_x_to_c(x_con, mask_con) + self.cat_x_to_c(x_cat, mask_cat)
        return self.con_x_to_c(x, mask)

    def predict(self, x, mask):
        z_mu, z_sig = self.c_to_zdist(self.x_to_c(x, mask))
        z = z_mu + z_sig * torch.randn_like(z_mu)
        return F.softmax(self.h_to_y(self.z_to_h(z)), dim=-1)

    def beta(self):
        # This allows us to carry out Cyclical Beta annealing if desired.
        # See: https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
        # for more info.
        # return min(1.0, 2*(self.epoch_count%20)/20)
        return 1.0

    def loss_func(self, x, y, mask, data_mask=None):
        mask_y = torch.bernoulli(torch.full_like(y.float(), 0.5))
        z_mu, z_sig = self.c_to_zdist(
            self.x_to_c(x, mask) + mask_y.unsqueeze(-1) * self.E_y[y]
        )
        hidden = self.z_to_h(z_mu + z_sig * torch.randn_like(z_mu))
        nll_x = self.nll_f(x, self.h_to_x(hidden), mask)
        nll_y = torch.mean(
            F.cross_entropy(self.h_to_y(hidden), y, reduction="none") * mask_y
        )
        return nll_x + nll_y + self.beta() * kl_01_loss(z_mu, z_sig)

    def calc_val_dict(self, val_loader, metric_f):
        val_metric = 0
        elbo = 0
        kl = 0
        nll_x = 0
        nll_y = 0
        for x, y, m_data in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            m_data = m_data.to(self.device)

            # Predict y based only on x.
            c_x = self.x_to_c(x, m_data)
            z_mu_x, z_sig_x = self.c_to_zdist(c_x)
            preds = F.softmax(
                self.h_to_y(self.z_to_h(z_mu_x + z_sig_x * torch.randn_like(z_mu_x))),
                dim=-1,
            )
            val_metric += metric_f(preds, y) / len(val_loader)

            # Calculate ELBO.
            z_mu, z_sig = self.c_to_zdist(c_x + self.E_y[y])
            hidden = self.z_to_h(z_mu + z_sig * torch.randn_like(z_mu))
            nll_x_tmp = self.nll_f(x, self.h_to_x(hidden), m_data)
            nll_y_tmp = F.cross_entropy(self.h_to_y(hidden), y)
            kl_tmp = kl_01_loss(z_mu, z_sig)
            elbo -= (nll_x_tmp + nll_y_tmp + kl_tmp) / len(val_loader)
            nll_x += nll_x_tmp / (len(val_loader) * self.num_features)
            nll_y += nll_y_tmp / len(val_loader)
            kl += kl_tmp / (len(val_loader) * self.latent_dim)
        return elbo, {
            "Predictive Metric": val_metric,
            "KL": kl,
            "NLL X": nll_x,
            "NLL Y": nll_y,
        }

    def calculate_acquisition_scores_subset(self, x, mask):
        with torch.no_grad():
            scores = torch.zeros_like(mask)
            batchsize = x.shape[0]

            x = torch.repeat_interleave(x, self.num_acquisition_samples, dim=0)
            mask = torch.repeat_interleave(mask, self.num_acquisition_samples, dim=0)
            c_xo = self.x_to_c(x, mask)
            z_mu_xo, z_sig_xo = self.c_to_zdist(c_xo)

            hidden = self.z_to_h(z_mu_xo + z_sig_xo * torch.randn_like(z_mu_xo))
            x_samples = self.sample_from_recon(self.h_to_x(hidden))
            y_samples = torch.multinomial(
                F.softmax(self.h_to_y(hidden), dim=-1), num_samples=1
            ).view(-1)
            del hidden  # Delete and clear as much as possible.

            for feature in range(mask.shape[-1]):
                self.clear_cache()
                x_io = x.clone()
                mask_tmp = mask.clone()
                mask_tmp[:, feature] = 1.0
                x_io[:, feature] = x_samples[:, feature]

                c_xio = self.x_to_c(x_io, mask_tmp)
                z_mu_xio, z_sig_xio = self.c_to_zdist(c_xio)
                kl1 = kl_div(z_mu_xio, z_sig_xio, z_mu_xo, z_sig_xo).view(
                    batchsize, self.num_acquisition_samples
                )
                kl1 = torch.mean(kl1, dim=-1)

                c_y = self.E_y[y_samples]
                z_mu_oy, z_sig_oy = self.c_to_zdist(c_xo + c_y)
                z_mu_oiy, z_sig_oiy = self.c_to_zdist(c_xio + c_y)
                kl2 = kl_div(z_mu_oiy, z_sig_oiy, z_mu_oy, z_sig_oy).view(
                    batchsize, self.num_acquisition_samples
                )
                kl2 = torch.mean(kl2, dim=-1)

                scores[:, feature] = kl1 - kl2
            return scores

    def calculate_acquisition_scores(self, x, mask):
        # We split the validation set if it is too large, since this
        # is an extremely memory intensive operation.
        with torch.no_grad():
            if x.nelement() * self.num_acquisition_samples <= self.size_limit:
                ids = [torch.arange(x.shape[0])]
            else:
                ids = np.array_split(
                    np.arange(x.shape[0]),
                    ceil(x.nelement() * self.num_acquisition_samples / self.size_limit),
                )
            scores = []
            for i in ids:
                # Clear cache before and after this, since it is memory intensive.
                self.clear_cache()
                scores.append(self.calculate_acquisition_scores_subset(x[i], mask[i]))
                self.clear_cache()
            scores = torch.cat(scores, dim=0)
            return scores - scores.min()  # Make sure all scores are greater than zero.
