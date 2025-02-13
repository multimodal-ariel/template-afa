# %%
from __future__ import annotations

import itertools as itrtls
from typing import Any, Callable, Literal

import mydatasets.aaco
import mymodels.classifiers
import numpy as np
import pandas as pd
import sklearn.base as skl_base
import tensordict as thd
import torch as th
import torchmetrics as thm


# %%
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


# %%
def make_templates(
    tdata: thd.TensorDict,
    is_classification: bool,
    max_features: int,
    top_k_features: int,
    feature_importance_type: Literal["eb"] = "eb",
    feature_importance_kwargs: dict[str, Any] = dict(),
) -> tuple[list[tuple[int, ...]], int]:
    txs: th.Tensor = tdata["xs"]
    tys: th.Tensor = tdata["ys"]
    n_covs: int = txs.shape[1]
    assert 1 < top_k_features and top_k_features <= n_covs
    assert 1 < max_features and max_features <= top_k_features
    # compute feature importances
    model: skl_base.BaseEstimator
    feature_importance_func: Callable[[], th.Tensor]
    if feature_importance_type == "dt":
        import sklearn.tree as skl_tree

        _ModelT = (
            skl_tree.DecisionTreeClassifier
            if is_classification
            else skl_tree.DecisionTreeRegressor
        )
        model = _ModelT(**feature_importance_kwargs)
        feature_importance_func = lambda: th.as_tensor(  # type: ignore  # noqa: E731
            model.feature_importances_, dtype=th.float32
        )
    elif feature_importance_type == "eb":
        import interpret.glassbox

        _ModelT = (
            interpret.glassbox.ExplainableBoostingClassifier
            if is_classification
            else interpret.glassbox.ExplainableBoostingRegressor
        )
        model = _ModelT(**feature_importance_kwargs)
        feature_importance_func = lambda: th.as_tensor(  # noqa: E731
            model.term_importances(), dtype=th.float32
        )
    else:
        raise ValueError(
            f"{feature_importance_type} feature_importance_type not supported"
        )
    model.fit(txs.numpy(), tys.numpy())
    fscores: th.Tensor = feature_importance_func()
    # ident top-k features and the best feature
    top_fidxs: th.Tensor = th.argsort(fscores, descending=True)
    best_fidx: int = int(top_fidxs[0].item())
    top_fidxs = top_fidxs[1:top_k_features]
    # make templates
    allfcombs_l: list[tuple[int, ...]] = list(
        itrtls.chain(
            *[
                itrtls.combinations(top_fidxs.tolist(), i)
                for i in range(0, max_features + 1)
            ]
        )
    )
    allfcombs_l = [tuple(sorted([best_fidx, *fcomb])) for fcomb in allfcombs_l]
    return allfcombs_l, best_fidx


# %%
def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
) -> tuple[th.Tensor, tuple[int, ...]]:
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    # repeat feature acquisition until all features in template has been acquired.
    for _ in itrtls.count():
        # randomly choose an initial template
        _fcomb_idx: int = int(th.randint(0, len(allfcombs_l), size=(1,)).item())
        _tmpl_fcomb: tuple[int, ...] = allfcombs_l[_fcomb_idx]
        # ident. unacquired features
        _tmp_fcomb: list[int] = [fidx for fidx in _tmpl_fcomb if fidx not in fobsd_l]
        if len(_tmp_fcomb) == 0:
            fcomb = _tmpl_fcomb
            break
        # randomly choose a feature to acquire
        _tmp_fcomb_idx = int(th.randint(0, len(_tmp_fcomb), size=(1,)).item())
        # add acquired feature to fcomb
        fobsd_l.append(_tmp_fcomb[_tmp_fcomb_idx])
        # terminate acq. if all features in template has been satisfied
        if all([fidx in fobsd_l for fidx in _tmpl_fcomb]):
            fcomb = _tmpl_fcomb
            break
    assert fcomb is not None
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fcomb] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :], acts)
    return pyhats[0], fcomb


# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("cube_20_0.3", to_normalize=False)
n_labels: int = len(th.unique(tdata["ys"]))

# %%
allfcombs_l, init_fidx = make_templates(
    tdata, is_classification=True, max_features=5, top_k_features=10
)

# %%
classifier = SubsetFeatureNaiveBayes(
    0.3, xs_train=tdata["xs"].numpy(), ys_train=tdata["ys"].numpy()
)
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)

# %%
for _data in vdata:
    _pyhat, _ = run_one_episode(
        x=_data["xs"],
        classifier=classifier,
        init_fidx=init_fidx,
        allfcombs_l=allfcombs_l,
    )
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
print(pd.Series(metrics_d))

# %%
