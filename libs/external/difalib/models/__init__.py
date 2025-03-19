from . import (
    VAEAC,
    actor_critic,
    data_utils,
    dataset,
    difa,
    diff_wrapper,
    eddi_wrapper,
    env,
    layers,
    model_utils,
    network_utils,
    ppo,
    prob_utils,
    random_wrapper,
    rl_wrapper,
    utils,
    vae_wrapper,
)

REGISTERED_MODELS = {
    "vaeac": vae_wrapper.Model,
    "eddi": eddi_wrapper.Model,
    "gsmrl": rl_wrapper.Model,
    "jafa": rl_wrapper.Model,
    "difa": diff_wrapper.Model,
    "random": random_wrapper.Model,
}
