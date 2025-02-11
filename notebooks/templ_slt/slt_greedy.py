# %%
from __future__ import annotations

import itertools as itrtls
import math
from typing import Any, Callable, Literal, TypedDict

import lightning as pl
import mydatasets.aaco
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import numpy as np
import pandas as pd
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm


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
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_templs: int,
    n_cands: int,
    lmbda: float,
    max_features: int,
) -> th.Tensor:
    txs: th.Tensor = tdata["xs"]
    tys: th.Tensor = tdata["ys"]
    n_covs: int = txs.shape[1]
    tmpls: th.Tensor = th.zeros((n_templs, n_covs), dtype=th.long)
    tmpl_fcs: list[tuple[int, ...]] = list()
    for _i in tqdm.trange(
        n_templs, desc="make templates", leave=True, dynamic_ncols=True
    ):
        # make candidate pool
        _ctmpl_fcs: list[tuple[int, ...]] = list()
        while True:
            if len(_ctmpl_fcs) >= n_cands:
                break
            _nfeats: int = int(th.randint(1, max_features + 1, (1,)).item())
            _fc_l: list[int] = th.multinomial(
                th.ones((n_covs,)), num_samples=_nfeats
            ).tolist()
            # make sure init feature is in fcomb
            if init_fidx not in _fc_l:
                _fc_l.append(init_fidx)
                # ensure max_features is kept
                if len(_fc_l) > max_features:
                    _fc_l = _fc_l[1:]
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc: tuple[int, ...] = tuple(_fc_l)
            if _fc not in _ctmpl_fcs and _fc not in tmpl_fcs:
                _ctmpl_fcs.append(tuple(_fc))
        _ctmpls: th.Tensor = th.zeros((n_cands, n_covs), dtype=th.long)
        for _j, _fc in enumerate(_ctmpl_fcs):
            _ctmpls[_j, _fc] = 1
        # classifier output
        _ctxs: th.Tensor = txs[:, None, :].expand(-1, n_cands, -1)
        _acts: th.Tensor = _ctmpls[None, :, :].expand(len(txs), -1, -1)
        _pyhats: th.Tensor = classifier.predict_proba(
            _ctxs.flatten(0, 1), _acts.flatten(0, 1)
        )
        _ys: th.Tensor = tys[:, None].expand(-1, n_cands).flatten(0, 1)
        # compute fitness function
        # (n_data, n_cands)
        _cels: th.Tensor = th.nn.functional.cross_entropy(
            _pyhats, _ys, reduction="none"
        ).unflatten(0, (len(txs), n_cands))
        # maximize fitness function
        # (n_cands, )
        _fitns: th.Tensor = th.mean(-_cels - lmbda * th.sum(_acts, dim=2), dim=0)
        _fcidx: int = int(th.argmax(_fitns).item())
        # update identified fcomb
        tmpl_fcs.append(_ctmpl_fcs[_fcidx])
        tmpls[_i, tmpl_fcs[-1]] = 1
    return tmpls


# %%
tcube, vcube = mydatasets.aaco.load_aaco_data("cube_20_0.3", to_normalize=False)
n_covs: int = tcube["xs"].shape[1]
n_labels: int = len(th.unique(tcube["ys"]))

# %%
classifier = SubsetFeatureNaiveBayes(
    0.3, xs_train=tcube["xs"].numpy(), ys_train=tcube["ys"].numpy()
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

#  %%
init_fidx: int = 6
n_templs: int = 256
n_cands: int = 64
lmbda: float = 0.1
max_features: int = 5

# %%
tmpls: th.Tensor = make_templates(
    tdata=tcube,
    classifier=classifier,
    init_fidx=init_fidx,
    n_templs=n_templs,
    n_cands=n_cands,
    lmbda=lmbda,
    max_features=max_features,
)

# %%
