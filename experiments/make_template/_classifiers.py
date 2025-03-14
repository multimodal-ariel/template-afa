from __future__ import annotations
import os

import hydra as hd
import mylib
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import numpy as np
import torch as th
from omegaconf import OmegaConf


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


def make_concat_nnet_classifier_from_pretrain_run(
    run_p: str,
    xs_train: np.ndarray,
    ys_train: np.ndarray,
    fit_kwargs: mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier._FitKwargs,
) -> mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier:
    # try to load file from project root if run_p does not exist
    if not os.path.exists(run_p):
        run_p = os.path.join(mylib.utils.get_project_root_dir(), run_p)
    run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
    n_covs: int = xs_train.shape[1]
    n_labels: int = len(np.unique(ys_train))
    classifier = mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
        nnet=hd.utils.instantiate(
            run_cfg.nnet,
            in_features=n_covs * 2,
            out_features=n_labels,
        ),
        xs_train=xs_train,
        ys_train=ys_train,
        fit_kwargs=fit_kwargs,
        state_dict_p=os.path.join(run_p, "classifier.pt"),
    )
    return classifier
