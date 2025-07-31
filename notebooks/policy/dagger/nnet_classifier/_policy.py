from __future__ import annotations

import numpy as np


class SKLClassifierPolicy:
    def __init__(self, sklc, n_covs: int):
        self.sklc, self.n_covs = sklc, n_covs

    def act(self, state_vec, training=False, epsilon=0.1) -> int:

        probs = self.sklc.predict_proba(state_vec.unsqueeze(0).cpu().numpy())[0]

        best = int(np.argmax(probs))
        return best
