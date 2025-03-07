from __future__ import annotations

import math
from typing import Any, Optional

import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import numpy as np
import torch as th
import tqdm.auto as tqdm
import xgboost as xgbst


class SubsetFeatureNaiveBayes(mymodels.classifiers.SubsetFeatureClassifier[None]):
    std: float

    def __init__(self, std: float, xs_train: np.ndarray, ys_train: np.ndarray):
        super().__init__(n_experts_per_act=1, xs_train=xs_train, ys_train=ys_train)
        self.std = std

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        pyhats: th.Tensor = self._aaco_forward_impl(
            ctxs.to(device="cpu"), acts.to(device="cpu")
        ).to(device=self.device)
        return pyhats

    def _aaco_forward_impl(self, x: th.Tensor, mask: th.Tensor):
        from scipy.stats import norm

        y_classes = list(range(self.n_labels))

        output_probs = th.zeros((len(x), self.n_labels))

        for y_val in y_classes:

            ## PDF values for each feature in x conditioned on the given label y_val

            # Default to PDF for U[0,1)
            p_x_y = th.where((x >= 0) & (x < 1), th.ones(x.shape), th.zeros(x.shape))

            # Use normal distribution PDFs for appropriate features given y_val
            p_x_y[:, y_val : y_val + 3] = th.transpose(
                th.Tensor(
                    np.array(
                        [
                            norm.pdf(x[:, y_val], y_val % 2, self.std),
                            norm.pdf(x[:, y_val + 1], (y_val // 2) % 2, self.std),
                            norm.pdf(x[:, y_val + 2], (y_val // 4) % 2, self.std),
                        ]
                    )
                ),
                0,
                1,
            )

            # Compute joint probability over masked features
            p_xo_y = th.prod(
                th.where(th.gt(mask, 0), p_x_y, th.tensor(1).float()), dim=1
            )

            p_y = 1 / self.n_labels

            output_probs[:, y_val] = p_xo_y * p_y

        return th.divide(
            output_probs,
            th.squeeze(th.dstack([th.sum(output_probs, dim=1)] * self.n_labels)),
        )

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None


class SubsetFeatureConcatXGBClassifier(
    mymodels.classifiers.SubsetFeatureConcatClassifier[None]
):
    xgb_kwargs: dict[str, Any]
    fraction_training_data_per_split: float
    n_splits: int
    n_tmpl_per_instance: int
    rseed: Optional[int]

    _models: list[xgbst.XGBClassifier]

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        xgb_kwargs: dict[str, Any],
        fraction_training_data_per_split: float,
        n_splits: int,
        n_tmpl_per_instance: int,
        rseed: Optional[int] = None,
    ):
        super().__init__(n_experts_per_act=1, xs_train=xs_train, ys_train=ys_train)
        self.xgb_kwargs = xgb_kwargs
        self.fraction_training_data_per_split = fraction_training_data_per_split
        self.n_splits = n_splits
        self.n_tmpl_per_instance = n_tmpl_per_instance
        self.rseed = rseed
        self._models = [xgbst.XGBClassifier(**self.xgb_kwargs) for _ in range(n_splits)]

    def fit_(self, acts_tmpls: th.Tensor) -> dict[str, float]:
        txs: th.Tensor = th.as_tensor(self.xs_train, dtype=th.float32)
        tys: th.Tensor = th.as_tensor(self.ys_train)
        pbar = tqdm.tqdm(
            self._models, desc="model rsplit", dynamic_ncols=True, leave=False
        )
        for _m in pbar:
            _n_data: int = math.ceil(len(txs) * self.fraction_training_data_per_split)
            _idxs: th.Tensor = th.randint(0, len(txs), size=(_n_data,), dtype=th.long)
            # (_n_data, n_tmpl_per_instance, n_covs)
            _xs: th.Tensor = txs[_idxs, None, :].expand(
                -1, self.n_tmpl_per_instance, -1
            )
            _fms: th.Tensor = th.stack(
                [
                    acts_tmpls[
                        th.multinomial(
                            th.arange(0, len(acts_tmpls), dtype=th.float32),
                            self.n_tmpl_per_instance,
                        )
                    ]
                    for _ in range(len(_xs))
                ]
            )
            # (_n_data, n_tmple_per_instance, 2 * n_covs)
            _minps: th.Tensor = th.cat((_xs * _fms, _fms), dim=2)
            # (_n_data, n_tmpl_per_instance)
            _ys: th.Tensor = tys[_idxs, None].expand(-1, self.n_tmpl_per_instance)
            _m.fit(_minps.flatten(0, 1).numpy(), _ys.flatten(0, 1).numpy())
        pbar.close()
        return dict()

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        device: th.device = ctxs.device
        ctxs = ctxs.to(device="cpu")
        acts = acts.to(device="cpu")
        # (n, n_covs * 2)
        minps: th.Tensor = th.cat((ctxs * acts, acts), dim=1)
        # (n, n_splits, n_labels)
        pyhats: th.Tensor = th.stack(
            [
                th.as_tensor(
                    _m.predict_proba(minps.numpy(force=True)), dtype=th.float32
                )
                for _m in self._models
            ],
            dim=1,
        )
        # (n, n_labels)
        pyhats = th.mean(pyhats, dim=1).to(device=device)
        return pyhats

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None
