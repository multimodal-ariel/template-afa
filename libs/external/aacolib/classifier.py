import os
from typing import TYPE_CHECKING

import hydra as hd
import mydatasets
import mylib
import mymodels
import numpy as np
import torch as th
from omegaconf import OmegaConf
from scipy.stats import norm
from xgboost import XGBClassifier


class NaiveBayes(th.nn.Module):
    def __init__(self, num_features, num_classes, std):
        super(NaiveBayes, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.std = std

    def forward(self, x):

        try:
            mask = x[:, self.num_features :]
            x = x[:, : self.num_features]
        except IndexError:
            raise Exception(
                "Classifier expects masking information to be concatenated with each feature vector."
            )

        y_classes = list(range(self.num_classes))

        output_probs = th.zeros((len(x), self.num_classes))

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

            p_y = 1 / self.num_classes

            output_probs[:, y_val] = p_xo_y * p_y

        return th.divide(
            output_probs,
            th.squeeze(th.dstack([th.sum(output_probs, axis=1)] * self.num_classes)),
        )

    def predict(self, x):
        return self.forward(x)


class classifier_xgb_dict:
    def __init__(self, output_dim, input_dim, subsample_ratio, X_train, y_train):
        """
        Input:
        output_dim: Dimension of the outcome y
        input_dim: Dimension of the input features (X)
        subsample_ratio: Fraction of training points for each boosting iteration

        Output:
        A dictionary of classifiers, predicting probabilities over y classes
        """
        self.xgb_model_dict = {}
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.subsample_ratio = subsample_ratio
        self.X_train_numpy = X_train.numpy()
        self.y_train_numpy = y_train.numpy()

    def __call__(self, X, idx):
        n = X.shape[0]
        probs = th.zeros((n, self.output_dim))
        for i in range(n):
            # Which mask?
            mask_i = X[i][self.input_dim :]
            nonzero_i = mask_i.nonzero().squeeze()
            mask_i_string = "".join(map(str, mask_i.long().tolist()))
            # Is the mask in the dictionary?
            if mask_i_string not in self.xgb_model_dict:
                self.xgb_model_dict[mask_i_string] = XGBClassifier(
                    n_estimators=250, max_depth=5, random_state=29, n_jobs=-1
                )
                X_train_subset = self.X_train_numpy[:, nonzero_i].reshape(
                    self.X_train_numpy.shape[0], -1
                )
                idx = np.random.choice(
                    X_train_subset.shape[0],
                    int(X_train_subset.shape[0] * self.subsample_ratio),
                    replace=False,
                )
                self.xgb_model_dict[mask_i_string].fit(
                    X_train_subset[idx], self.y_train_numpy[idx]
                )
            # Prediction
            probs[i] = th.from_numpy(
                self.xgb_model_dict[mask_i_string].predict_proba(
                    X[i, nonzero_i].numpy().reshape(1, -1)
                )
            )
        return probs


class classifier_ground_truth:
    def __init__(self, num_features=20, num_classes=8, std=0.3):
        self.gt_classifier = NaiveBayes(
            num_features=num_features, num_classes=num_classes, std=std
        )

    def __call__(self, X, idx):
        return self.gt_classifier.predict(X)


class classifier_xgb:
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        self.xgb_model.set_params(n_jobs=-1)

    def __call__(self, X, idx):
        return th.tensor(self.xgb_model.predict_proba(X))


def _make_concat_nnet_classifier_from_pretrain_run(
    run_p: str,
    xs_train: np.ndarray,
    ys_train: np.ndarray,
    fit_kwargs: mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier._FitKwargs,
    classifier_fn: str = "classifier.pt",
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
        state_dict_p=os.path.join(run_p, classifier_fn),
    )
    return classifier


class CubeNeuralNetClassifier:

    def __init__(self):
        tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("cube_20_0.3")
        self.nnet = _make_concat_nnet_classifier_from_pretrain_run(
            run_p="experiments/pretrain/nnet-random-subset/outputs/cube/20251211_214511",
            xs_train=tdata["xs"],
            ys_train=tdata["ys"],
            fit_kwargs={
                "opt_type": th.optim.Adam,
                "opt_kwargs": {"lr": 1e03},
                "n_iter": 100,
                "bsz": 4096,
            },
        )
        self.nnet

    def __call__(self, X, idx):
        return self.nnet.predict_proba(*th.chunk(X, chunks=2, dim=1))
