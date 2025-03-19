from operator import itemgetter

import torch
import torch.nn as nn

from .actor_critic import Actor, ActorCritic
from .layers import (
    ImageModels,
    ImagePolicyEncoder,
    InputLayer,
    MemoryLayer,
    MixedInputLayer,
    SkipConnection,
    VAEImageInputLayer,
    VAEImageOutputLayer,
)
from .prob_utils import GaussianCategoricalLoss, ImageGaussianCategoricalLoss
from .utils import get_real_cat_features
from .VAEAC import VAEAC


def get_pred_network_cnn(params, data_parameters):
    return ImageModels(params, data_parameters)


def get_ac_network_cnn(params, data_parameters):
    shared_layers = ImagePolicyEncoder(params, data_parameters)
    n_features = data_parameters["n_features"]
    width = params.hidden_dim
    actor_layer = nn.Linear(width, n_features + 1)
    if params.problem == "difa":
        return Actor(shared_layers, actor_layer)
    value_layer = nn.Linear(width, 1)
    return ActorCritic(shared_layers, actor_layer, value_layer)


def get_pred_network(params, data_parameters):
    if params.cnn:
        return get_pred_network_cnn(params, data_parameters)
    real_features, cat_features, cat_categories = get_real_cat_features(
        data_parameters, add_label=False
    )

    width, depth, dropout, embed_dim = (
        params.hidden_dim,
        params.depth,
        params.dropout,
        params.embed_dim,
    )
    shared_input_layer = InputLayer(
        MixedInputLayer(data_parameters, width, embed_dim, add_label=False)
    )
    input_dim = (
        params.embed_dim * len(data_parameters["categorical_classes"])
        + (
            params.hidden_dim
            if len(data_parameters["categorical_classes"])
            < data_parameters["n_features"]
            else 0
        )
        + data_parameters["n_features"]
    )
    shared_hidden_layer = nn.Linear(input_dim, width)
    pred_layers = [
        shared_input_layer,
        shared_hidden_layer,
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    ]
    for i in range(depth):
        pred_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )

    pred_layers.append(nn.Linear(width, data_parameters["n_classes"]))
    pred_model = nn.Sequential(*pred_layers)
    return pred_model


def get_pred_policy_network_cnn(params, data_parameters):
    pred_model = get_pred_network_cnn(params, data_parameters)
    ac_network = get_ac_network_cnn(params, data_parameters)
    return pred_model, ac_network


def get_pred_policy_network(params, data_parameters):
    if params.cnn:
        return get_pred_policy_network_cnn(params, data_parameters)
    real_features, cat_features, cat_categories = get_real_cat_features(
        data_parameters, add_label=False
    )

    width, depth, dropout, embed_dim = (
        params.hidden_dim,
        params.depth,
        params.dropout,
        params.embed_dim,
    )
    n_features = len(real_features) + len(cat_features)
    class_dim = 2 if data_parameters["n_classes"] <= 1 else data_parameters["n_classes"]
    shared_input_layer = InputLayer(
        MixedInputLayer(data_parameters, width, embed_dim, add_label=False)
    )
    input_dim = (
        params.embed_dim * len(data_parameters["categorical_classes"])
        + (
            params.hidden_dim
            if len(data_parameters["categorical_classes"])
            < data_parameters["n_features"]
            else 0
        )
        + data_parameters["n_features"]
    )
    if params.use_aux_state:
        input_dim += n_features * 3 + class_dim

    shared_hidden_layer = nn.Linear(input_dim, width)
    pred_layers = [
        shared_input_layer,
        shared_hidden_layer,
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    ]
    for i in range(depth):
        pred_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )

    pred_layers.append(nn.Linear(width, data_parameters["n_classes"]))
    pred_model = nn.Sequential(*pred_layers)
    if params.nosharing:
        ac_input_layer = InputLayer(
            MixedInputLayer(data_parameters, width, embed_dim, add_label=False)
        )
    else:
        ac_input_layer = shared_input_layer
    ac_hidden_layer = nn.Linear(input_dim, width)
    shared_layers = [
        ac_input_layer,
        ac_hidden_layer,
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    ]
    for i in range(depth):
        shared_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )

    shared_layers = nn.Sequential(*shared_layers)

    actor_layer = nn.Linear(width, n_features + 1)
    value_layer = nn.Linear(width, 1)
    if params.problem == "difa":
        return pred_model, Actor(shared_layers, actor_layer)
    return pred_model, ActorCritic(shared_layers, actor_layer, value_layer)


def get_imputation_model(params, data_parameters):
    keys = [
        "rec_log_prob",
        "proposal_network",
        "prior_network",
        "generative_network",
    ]
    (
        rec_log_prob,
        proposal_network,
        prior_network,
        generative_network,
    ) = itemgetter(
        *keys
    )(build_vae_blocks(params, data_parameters))
    model = VAEAC(rec_log_prob, proposal_network, prior_network, generative_network)
    return model


def build_vae_blocks_cnn(params, data_parameters):
    # Proposal Network
    proposal_network = VAEImageInputLayer(params, data_parameters)
    prior_network = VAEImageInputLayer(params, data_parameters)
    # Generative network
    generative_network = VAEImageOutputLayer(params, data_parameters)
    return {
        "rec_log_prob": ImageGaussianCategoricalLoss(
            data_parameters, min_sigma=params.min_sigma
        ),
        "proposal_network": proposal_network,
        "prior_network": prior_network,
        "generative_network": generative_network,
    }


def build_vae_blocks(params, data_parameters):
    if params.cnn:
        return build_vae_blocks_cnn(params, data_parameters)
    width, depth, dropout, latent_dim, embed_dim = (
        params.hidden_dim,
        params.depth,
        params.dropout,
        params.latent_dim,
        params.embed_dim,
    )
    input_dim = (
        params.embed_dim * len(data_parameters["categorical_classes"])
        + (
            params.hidden_dim
            if len(data_parameters["categorical_classes"])
            < data_parameters["n_features"]
            else 0
        )
        + (params.embed_dim if data_parameters["n_classes"] > 1 else 0)
        + 1
        + data_parameters["n_features"]
    )
    output_dim = (
        data_parameters["n_features"] * 2
        + sum(data_parameters["categorical_classes"].values())
        - len(data_parameters["categorical_classes"]) * 2
        + max(2, data_parameters["n_classes"])
    )
    # Proposal Network
    proposal_layers = [
        InputLayer(MixedInputLayer(data_parameters, width, embed_dim, add_label=True)),
        nn.Linear(input_dim, width),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    ]
    for i in range(depth):
        proposal_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    proposal_layers.append(nn.Linear(width, latent_dim * 2))
    proposal_network = nn.Sequential(*proposal_layers)
    # Prior Network
    prior_layers = [
        InputLayer(MixedInputLayer(data_parameters, width, embed_dim, add_label=True)),
        MemoryLayer("#input"),
        nn.Linear(input_dim, width),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    ]
    for i in range(depth):
        prior_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer("#%d" % i),
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    prior_layers.extend(
        [
            MemoryLayer("#%d" % depth),
            nn.Linear(width, latent_dim * 2),
        ]
    )
    prior_network = nn.Sequential(*prior_layers)
    # Generative network
    generative_layers = [
        nn.Linear(latent_dim, width),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    ]
    for i in range(depth + 1):
        generative_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer("#%d" % (depth - i), True),
                nn.Linear(width * 2, width),
                nn.LeakyReLU(),
            )
        )
    generative_layers.extend(
        [
            MemoryLayer("#input", True),
            nn.Linear(width + input_dim, output_dim),
        ]
    )
    generative_network = nn.Sequential(*generative_layers)
    return {
        "rec_log_prob": GaussianCategoricalLoss(
            data_parameters, min_sigma=params.min_sigma
        ),
        "proposal_network": proposal_network,
        "prior_network": prior_network,
        "generative_network": generative_network,
    }
