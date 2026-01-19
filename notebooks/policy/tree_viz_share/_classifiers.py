from __future__ import annotations

import math
import os
from typing import Any

import hydra as hd
import mydatasets
import mylib
import mymodels
import numpy as np
import pandas as pd
import sklearn.preprocessing as skl_preproc
import sklearn.tree as skl_tree
import torch as th
import tqdm.auto as tqdm
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


def make_concat_nnet_classifier_from_make_templates_run(
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
    classifier = make_concat_nnet_classifier_from_pretrain_run(
        run_cfg.tclassifier.run_p, xs_train, ys_train, fit_kwargs
    )
    classifier.load_state_dict(
        th.load(os.path.join(run_p, classifier_fn), map_location="cpu")
    )
    return classifier


class EngineFaultDecisionTreeClassifier(
    mymodels.classifiers.SubsetFeatureClassifier[skl_tree.DecisionTreeClassifier]
):
    FEATURE_NAMES = np.asarray(
        [
            "MAP",
            "TPS",
            "Force",
            "Power",
            "RPM",
            "Consumption L/H",
            "Consumption L/100KM",
            "Speed",
            "CO",
            "HC",
            "CO2",
            "O2",
            "Lambda",
            "AFR",
        ],
        dtype=str,
    )
    CLASS_NAMES = np.asarray(
        [
            "normal",
            "rich mixture",
            "lean mixture",
            "low voltage",
        ],
        dtype=str,
    )
    dtc_kwargs: dict[str, Any]
    stdsclr: skl_preproc.StandardScaler

    _inv_txs: np.ndarray

    def __init__(
        self,
        dtc_kwargs: dict[str, Any],
        xs_train: np.ndarray,
        ys_train: np.ndarray,
    ):
        super().__init__(n_experts_per_act=1, xs_train=xs_train, ys_train=ys_train)
        self.dtc_kwargs = dtc_kwargs
        # engine fault dataset is loaded with normalized feature
        # need a standard scaler on training data to invert things back
        self.stdsclr = skl_preproc.StandardScaler(copy=True).fit(
            mydatasets.aaco.load_aaco_data("engine-fault", to_normalize=False)[0][
                "xs"
            ].numpy(force=True)
        )  # type:ignore
        self._inv_txs = self.stdsclr.inverse_transform(self.xs_train)

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
        xs_df: pd.DataFrame = pd.DataFrame(xs, columns=self.FEATURE_NAMES[fcomb,])
        model.fit(xs_df, ys)
        self._act_to_classifier[key] = model
        return model


class EngineFaultXGBClassifier(mymodels.classifiers.SubsetFeatureConcatXGBClassifier):
    def __init__(
        self,
        xs_train: np.ndarray[tuple[Any, ...], np.dtype[Any]],
        ys_train: np.ndarray[tuple[Any, ...], np.dtype[Any]],
        xgb_kwargs: dict[str, Any],
        fraction_training_data_per_split: float,
        n_splits: int,
        n_tmpl_per_instance: int,
        rseed: int | None = None,
    ):
        super().__init__(
            xs_train,
            ys_train,
            xgb_kwargs,
            fraction_training_data_per_split,
            n_splits,
            n_tmpl_per_instance,
            rseed,
        )

    def fit_(self, acts_tmpls: th.Tensor | None) -> dict[str, float]:
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
            _fms: th.Tensor = (
                th.stack(
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
                if acts_tmpls is not None
                else th.randint_like(_xs, 0, 2, dtype=th.float32)
            )
            # # (_n_data, n_tmple_per_instance, n_covs)
            _minps: th.Tensor = th.where(_fms.to(dtype=th.bool), _xs, th.nan)
            # _minps: th.Tensor = th.cat((_xs * _fms, _fms), dim=2)
            # (_n_data, n_tmpl_per_instance)
            _ys: th.Tensor = tys[_idxs, None].expand(-1, self.n_tmpl_per_instance)
            _m.fit(_minps.flatten(0, 1).numpy(), _ys.flatten(0, 1).numpy())
        pbar.close()
        return dict()

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        device: th.device = ctxs.device
        ctxs = ctxs.to(device="cpu")
        acts = acts.to(device="cpu")
        # (n, n_covs)
        minps: th.Tensor = th.where(acts.to(dtype=th.bool), ctxs, th.nan)
        # minps: th.Tensor = th.cat((ctxs * acts, acts), dim=1)
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
