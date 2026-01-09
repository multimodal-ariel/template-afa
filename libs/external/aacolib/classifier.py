import os
from typing import Any

import hydra as hd
import mydatasets
import mylib
import mymodels
import numpy as np
import sklearn.preprocessing as skl_preproc
import sklearn.tree as skl_tree
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


class EngineFaultDecisionTreeClassifier(
    mymodels.classifiers.SubsetFeatureClassifier[skl_tree.DecisionTreeClassifier]
):
    dtc_kwargs: dict[str, Any]
    stdsclr: skl_preproc.StandardScaler

    _inv_txs: np.ndarray

    def __init__(self):
        tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(
            "engine-fault", to_normalize=True
        )
        super().__init__(
            n_experts_per_act=1,
            xs_train=tdata["xs"].numpy(force=True),
            ys_train=tdata["ys"].numpy(force=True),
        )
        self.dtc_kwargs = {
            "max_depth": 8,
            "splitter": "best",
            "criterion": "log_loss",
            "random_state": 279,
        }
        # engine fault dataset is loaded with normalized feature
        # need a standard scaler on training data to invert things back
        self.stdsclr = skl_preproc.StandardScaler(copy=True).fit(
            mydatasets.aaco.load_aaco_data("engine-fault", to_normalize=False)[0][
                "xs"
            ].numpy(force=True)
        )  # type:ignore
        self._inv_txs = self.stdsclr.inverse_transform(self.xs_train)

    def forward(self, X, idx):
        return self.predict_proba(*th.chunk(X, chunks=2, dim=1))

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        ctxs = th.as_tensor(
            self.stdsclr.inverse_transform(ctxs.numpy(force=True)),
            dtype=th.float32,
            device=ctxs.device,
        )
        n: int = len(ctxs)
        n_labels: int = self.n_labels
        acts = acts.to(dtype=th.long)
        acts_l: list[tuple[int, ...]] = [tuple(a.tolist()) for a in acts]
        pyhats: th.Tensor = th.empty((n, n_labels), dtype=th.float32)
        for act in set(acts_l):
            _act: th.Tensor = th.as_tensor([act], dtype=th.long, device=acts.device)
            _curr_idxs: th.Tensor = th.argwhere(th.all(acts == _act, dim=1)).flatten()
            _ctxs: th.Tensor = ctxs[_curr_idxs]
            pyhats[_curr_idxs] = self._predict_proba_same_act(_ctxs, act)
        return pyhats

    def _predict_proba_same_act(
        self, ctxs: th.Tensor, act: tuple[int, ...]
    ) -> th.Tensor:
        classifier = self[act]
        ctxs_: np.ndarray = ctxs.numpy(force=True)
        fcomb: tuple[int, ...] = self.act_to_fcomb_exidx(act)[0]
        pyhats_n: np.ndarray = classifier.predict_proba(ctxs_[:, fcomb])  # type:ignore
        pyhats: th.Tensor = th.as_tensor(pyhats_n, dtype=th.float32)
        return pyhats

    def __getitem__(self, key: tuple[int, ...]) -> skl_tree.DecisionTreeClassifier:
        assert self.n_experts_per_act == 1
        assert len(key) == self.n_covs
        if key in self._act_to_classifier:
            return self._act_to_classifier[key]
        act: tuple[int, ...] = key
        fcomb: tuple[int, ...] = self.act_to_fcomb_exidx(act)[0]
        model = skl_tree.DecisionTreeClassifier(**self.dtc_kwargs)
        xs: np.ndarray = self._inv_txs[:, fcomb].astype(np.float32)
        ys: np.ndarray = self.ys_train.astype(np.int64)
        model.fit(xs, ys)
        self._act_to_classifier[key] = model
        return model
