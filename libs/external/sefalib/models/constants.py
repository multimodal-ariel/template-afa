"""Model constants used throughout."""

import numpy as np


log_eps = 1e-8
min_sig = 1e-3
half_log_2pi = 0.5 * np.log(2 * np.pi)

# These are for LR Reduce on Plateau Schedulers.
cooldown = 0
lr_factor = 0.2
min_lr = 1e-7
