import torch
from torch.distributions import Categorical, Normal
from torch.nn import Module
from torch.nn.functional import softmax, softplus

from .utils import get_real_cat_features


class ImageGaussianCategoricalLoss(Module):
    """
    This layer computes log probability of groundtruth for each object
    given the mask and the distribution parameters.
    This layer works for the cases when the dataset contains both
    real-valued and categorical features.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.
    In the first case the distribution over feature is categorical,
    in the second case it is Gaussian.

    For example, if one_hot_max_sizes is [4, 1, 1, 2], then the distribution
    parameters for one object is the vector
    [p_00, p_01, p_02, p_03, mu_1, sigma_1, mu_2, sigma_2, p_30, p_31],
    where Softmax([p_00, p_01, p_02, p_03]) and Softmax([p_30, p_31])
    are probabilities of the first and the fourth feature categories
    respectively in the model generative distribution, and
    Gaussian(mu_1, sigma_1 ^ 2) and Gaussian(mu_2, sigma_2 ^ 2) are
    the model generative distributions on the second and the third features.

    For the definitions of min_sigma and min_prob see normal_parse_params
    and categorical_parse_params docs.

    This layer works correctly with missing values in groundtruth
    which are represented by NaNs.

    This layer works with 2D inputs only.
    """

    def __init__(self, data_parameters, min_sigma=1e-1, min_prob=1e-4):
        super().__init__()
        self.image_shape = data_parameters["shape"]
        self.n_classes = data_parameters["n_classes"]
        self.min_sigma = min_sigma
        self.min_prob = min_prob

    def forward(self, batch, distr_params, unobserved_mask):
        log_probs = []
        batch_masked = batch.clone()
        # image
        image_params = distr_params[0]  # B,2C,H,W
        mu = image_params[:, : self.image_shape[0], :, :]
        sigma_params = image_params[:, self.image_shape[0] :, :, :]
        sigma = softplus(sigma_params)
        sigma = sigma.clamp(min=self.min_sigma)
        distr = Normal(mu, sigma)

        groundtruth_col = batch_masked[:, :-1].view(len(batch), *self.image_shape)
        log_prob = distr.log_prob(groundtruth_col) * unobserved_mask[:, :-1].view(
            len(batch), *self.image_shape
        )
        log_probs.append(log_prob.view(len(batch), -1))
        # labels
        label_params = distr_params[1]
        distr = categorical_parse_params_column(label_params, self.min_prob)
        log_prob = (distr.log_prob(batch_masked[:, -1]) * unobserved_mask[:, -1])[
            :, None
        ]
        log_probs.append(log_prob)
        return torch.cat(log_probs, 1).sum(-1)

    def sampler(self, distr_params, sample_most_probable=False):
        # image
        image_params = distr_params[0]  # B,2C,H,W
        mu = image_params[:, : self.image_shape[0], :, :]
        sigma_params = image_params[:, self.image_shape[0] :, :, :]
        sigma = softplus(sigma_params)
        sigma = sigma.clamp(min=self.min_sigma)
        distr = Normal(mu, sigma)
        # labels
        label_params = distr_params[1]
        label_distr = categorical_parse_params_column(label_params, self.min_prob)
        if sample_most_probable:
            image = distr.mean.view(len(image_params), -1)
            label = torch.max(label_distr.probs, 1)[1].float().view(-1, 1)
        else:
            image = distr.sample().view(len(image_params), -1)
            label = label_distr.sample().view(-1, 1)
        samples = torch.cat([image, label], dim=-1)
        return samples

    def label_sampler(self, distr_params, sample_most_probable):
        # labels
        label_params = distr_params[1]
        distr = categorical_parse_params_column(label_params, self.min_prob)
        if sample_most_probable:
            label = torch.max(distr.probs, 1)[1].float().view(-1)
        else:
            label = distr.sample().view(-1)

        return label, distr

    def get_entropy_mean_std(self, distr_params, entropy, mean, std, only_features):
        entropies, means, stds = [], [], []
        # image
        image_params = distr_params[0]  # B,2C,H,W
        mu = image_params[:, : self.image_shape[0], :, :]
        sigma_params = image_params[:, self.image_shape[0] :, :, :]
        sigma = softplus(sigma_params)
        sigma = sigma.clamp(min=self.min_sigma)
        distr = Normal(mu, sigma)
        if not only_features:
            # labels
            label_params = distr_params[1]
            label_distr = categorical_parse_params_column(label_params, self.min_prob)
        if entropy:
            entropies.append(distr.entropy().view(len(image_params), -1))
            if not only_features:
                entropies.append(label_distr.entropy().view(-1, 1))
            entropies = torch.cat(entropies, dim=-1)
        if mean:
            means.append(distr.mean.view(len(image_params), -1))
            if not only_features:
                max_prob_vals = torch.max(label_distr.probs, 1)[1].float()
                means.append(max_prob_vals.view(-1, 1))
            means = torch.cat(means, dim=-1)
        if std:
            stds.append((distr.variance**0.5).view(len(image_params), -1))
            if not only_features:
                max_probs = torch.max(label_distr.probs, 1)[0]
                stds.append((max_probs * (1 - max_probs)).view(-1, 1) ** 0.5)
            stds = torch.cat(stds, dim=-1)
        return entropies, means, stds


class GaussianCategoricalLoss(Module):
    """
    This layer computes log probability of groundtruth for each object
    given the mask and the distribution parameters.
    This layer works for the cases when the dataset contains both
    real-valued and categorical features.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.
    In the first case the distribution over feature is categorical,
    in the second case it is Gaussian.

    For example, if one_hot_max_sizes is [4, 1, 1, 2], then the distribution
    parameters for one object is the vector
    [p_00, p_01, p_02, p_03, mu_1, sigma_1, mu_2, sigma_2, p_30, p_31],
    where Softmax([p_00, p_01, p_02, p_03]) and Softmax([p_30, p_31])
    are probabilities of the first and the fourth feature categories
    respectively in the model generative distribution, and
    Gaussian(mu_1, sigma_1 ^ 2) and Gaussian(mu_2, sigma_2 ^ 2) are
    the model generative distributions on the second and the third features.

    For the definitions of min_sigma and min_prob see normal_parse_params
    and categorical_parse_params docs.

    This layer works correctly with missing values in groundtruth
    which are represented by NaNs.

    This layer works with 2D inputs only.
    """

    def __init__(self, data_parameters, min_sigma=1e-1, min_prob=1e-4):
        super().__init__()
        (
            real_features,
            cat_features,
            cat_categories,
        ) = get_real_cat_features(data_parameters, add_label=True)
        self.n_features = len(real_features) + len(cat_features)
        self.feature_vals = {}
        for col in real_features:
            self.feature_vals[col] = 1
        for idx, col in enumerate(cat_features):
            self.feature_vals[col] = cat_categories[idx]

        self.min_sigma = min_sigma
        self.min_prob = min_prob

    def forward(self, batch, distr_params, mask):
        log_probs = []
        col_idx = 0
        nan_mask = torch.isnan(batch)
        batch_masked = batch.clone()
        batch_masked[nan_mask] = 0
        unobserved_mask = mask * (1.0 - nan_mask.float())
        for col in range(self.n_features):
            n_vals = self.feature_vals[col]
            groundtruth_col = batch_masked[:, col : col + 1]
            if n_vals == 1:
                params = distr_params[:, col_idx : col_idx + 2]
                col_idx += 2
                distr = normal_parse_params(params, self.min_sigma)
                log_prob = (
                    distr.log_prob(groundtruth_col) * unobserved_mask[:, col : col + 1]
                )
            else:
                params = distr_params[:, col_idx : col_idx + n_vals]
                col_idx += n_vals
                distr = categorical_parse_params_column(params, self.min_prob)
                log_prob = (
                    distr.log_prob(groundtruth_col.squeeze(1)) * unobserved_mask[:, col]
                )[:, None]
            log_probs.append(log_prob)
        return torch.cat(log_probs, 1).sum(-1)

    def sampler(self, distr_params, sample_most_probable=False):
        samples = torch.zeros(len(distr_params), self.n_features).to(
            distr_params.device
        )
        col_idx = 0
        for col in range(self.n_features):
            n_vals = self.feature_vals[col]
            if n_vals == 1:
                params = distr_params[:, col_idx : col_idx + 2]
                col_idx += 2
                distr = normal_parse_params(params, self.min_sigma)
            else:
                params = distr_params[:, col_idx : col_idx + n_vals]
                col_idx += n_vals
                distr = categorical_parse_params_column(params, self.min_prob)
            if sample_most_probable:
                if n_vals == 1:
                    col_samples = distr.mean
                else:
                    col_samples = torch.max(distr.probs, 1)[1].float()
            else:
                col_samples = distr.sample()
            samples[:, col : col + 1] = col_samples.view(-1, 1)
        return samples

    def label_sampler(self, distr_params, sample_most_probable):
        n_vals = self.feature_vals[self.n_features - 1]
        is_categorical = n_vals > 1
        if is_categorical:
            y_dist = categorical_parse_params_column(
                distr_params[:, -n_vals:], self.min_prob
            )
            if sample_most_probable:
                y_hat = torch.max(y_dist.probs, 1)[1].float()
            else:
                y_hat = y_dist.sample()
        else:
            y_dist = normal_parse_params(
                distr_params[:, -n_vals * 2 :], min_sigma=self.min_sigma
            )
            if sample_most_probable:
                y_hat = y_dist.mean.squeeze()
            else:
                y_hat = y_dist.sample().squeeze()

        return y_hat, y_dist

    def get_entropy_mean_std(self, distr_params, entropy, mean, std, only_features):
        entropies, means, stds, col_idx = [], [], [], 0
        n_features = self.n_features
        if only_features:
            n_features -= 1
        for col in range(n_features):
            n_vals = self.feature_vals[col]
            if n_vals == 1:
                params = distr_params[:, col_idx : col_idx + 2]
                col_idx += 2
                distr = normal_parse_params(params, self.min_sigma)
            else:
                params = distr_params[:, col_idx : col_idx + n_vals]
                col_idx += n_vals
                distr = categorical_parse_params_column(params, self.min_prob)
            if entropy:
                entropies.append(distr.entropy().view(-1, 1))
            if mean:
                if n_vals == 1:
                    means.append(distr.mean)
                else:
                    max_prob_vals = torch.max(distr.probs, 1)[1].float()
                    means.append(max_prob_vals.view(-1, 1))
            if std:
                if n_vals == 1:
                    stds.append(distr.variance**0.5)
                else:
                    max_probs = torch.max(distr.probs, 1)[0]
                    stds.append((max_probs * (1 - max_probs)).view(-1, 1) ** 0.5)
        if entropies:
            entropies = torch.cat(entropies, dim=-1)
        if means:
            means = torch.cat(means, dim=-1)
        if stds:
            stds = torch.cat(stds, dim=-1)
        return entropies, means, stds


def normal_parse_params(params, min_sigma=0):
    """
    Take a Tensor (e. g. neural network output) and return
    torch.distributions.Normal distribution.
    This Normal distribution is component-wise independent,
    and its dimensionality depends on the input shape.
    First half of channels is mean of the distribution,
    the softplus of the second half is std (sigma), so there is
    no restrictions on the input tensor.

    min_sigma is the minimal value of sigma. I. e. if the above
    softplus is less than min_sigma, then sigma is clipped
    from below with value min_sigma. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    d = params.shape[1]
    mu = params[:, : d // 2]
    sigma_params = params[:, d // 2 :]
    sigma = softplus(sigma_params)
    sigma = sigma.clamp(min=min_sigma)
    distr = Normal(mu, sigma)
    return distr


def categorical_parse_params_column(params, min_prob=0):
    """
    Take a Tensor (e. g. a part of neural network output) and return
    torch.distributions.Categorical distribution.
    The input tensor after applying softmax over the last axis contains
    a batch of the categorical probabilities. So there are no restrictions
    on the input tensor.

    Technically, this function treats the last axis as the categorical
    probabilities, but Categorical takes only 2D input where
    the first axis is the batch axis and the second one corresponds
    to the probabilities, so practically the function requires 2D input
    with the batch of probabilities for one categorical feature.

    min_prob is the minimal probability for each class.
    After clipping the probabilities from below they are renormalized
    in order to be a valid distribution. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    params = softmax(params, -1)
    params = params.clamp(min_prob)
    params = params / params.sum(-1, keepdim=True)
    distr = Categorical(probs=params)
    return distr
