# %%
from __future__ import annotations

import itertools as itrtls
from typing import Any, Callable, Literal, TypedDict

import lightning as pl
import mydatasets.aaco
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import numpy as np
import pandas as pd
import sklearn.base as skl_base
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
class SoftmaxSelector(th.nn.Module):
    n_covs: int
    n_templates: int
    nnet: th.nn.Module

    def __init__(self, n_covs: int, n_templates: int, nnet: th.nn.Module) -> None:
        super().__init__()
        self.n_covs = n_covs
        self.n_templates = n_templates
        self.nnet = nnet

    def forward(self, sinps: th.Tensor, to_probs: bool) -> th.Tensor:
        outs: th.Tensor = self.nnet(sinps)
        if to_probs:
            outs = th.nn.functional.softmax(outs)
        return outs


# %%
def fcomb_to_xact(n_covs: int, fcomb: tuple[int, ...]) -> th.Tensor:
    xact = th.zeros((n_covs), dtype=th.float32)
    xact[fcomb] = 1
    return xact


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


def run_one_random_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
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
    return pyhats[0], fobsd_l, fcomb


compute_reward = lambda pyhat, y: th.nn.functional.cross_entropy(  # noqa: E731
    pyhat[None, :], y[None], reduction="none"
)[0]


# %%
def make_init_queries(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
    fcomb_to_slbl: dict[tuple[int, ...], int],
    init_capital: int,
) -> thd.TensorDict:
    n_covs: int = tdata["xs"].shape[1]
    sinps: th.Tensor = th.zeros((init_capital, n_covs * 2), dtype=th.float32)
    ctxs: th.Tensor = sinps[:, :n_covs]
    mobs: th.Tensor = sinps[:, n_covs : n_covs * 2]
    xacts: th.Tensor = th.zeros((init_capital, tdata["xs"].shape[1]), dtype=th.float32)
    slbls: th.Tensor = th.empty((init_capital,), dtype=th.long)
    rewards: th.Tensor = th.empty((init_capital,), dtype=th.float32)
    for _i, _data in enumerate(tdata[th.randint(0, len(tdata), (init_capital,))]):
        _pyhat, _fobsd_l, _fcomb = run_one_random_episode(
            x=_data["xs"],
            classifier=classifier,
            init_fidx=init_fidx,
            allfcombs_l=allfcombs_l,
        )
        mobs[_i, _fobsd_l] = 1.0
        ctxs[_i] = _data["xs"] * mobs[_i]
        xacts[_i, _fcomb] = 1.0
        slbls[_i] = fcomb_to_slbl[_fcomb]
        rewards[_i] = compute_reward(_pyhat, _data["ys"])
    xps = thd.TensorDict(
        {"sinps": sinps, "slbls": slbls, "xacts": xacts, "rewards": rewards}
    ).auto_batch_size_(1)
    return xps


def _collate_fn(
    bxp: thd.TensorDict,
    n_covs: int,
    init_fidx: int,
) -> thd.TensorDict:
    bsz: int = bxp.batch_size[0]
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs))
    bnms[:, init_fidx] = 1
    bnms = th.cat((bnms, bnms), dim=1)
    bsinp: th.Tensor = bxp["sinps"] * bnms
    bnxp: thd.TensorDict = bxp.clone()
    bnxp["sinps"] = bsinp
    return bnxp


class _TrainState(TypedDict):
    selector: SoftmaxSelector
    opt: th.optim.Optimizer
    n_trial_itr: int
    n_fit_itr: int
    opt_step: int


def _fit_iter(
    tstate: _TrainState,
    tloader: th_data.DataLoader,
    pbar: tqdm.tqdm,
    plf: pl.Fabric,
) -> dict[str, float]:
    selector: SoftmaxSelector = tstate["selector"].to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    slosses_l: list[th.Tensor] = list()
    for bxps in tloader:
        bxps = bxps.to(device=plf.device)
        bsinps: th.Tensor = bxps["sinps"]
        bslbls: th.Tensor = bxps["slbls"]
        bsouts: th.Tensor = selector(bsinps, to_probs=False)
        # compute selector loss
        bslosses: th.Tensor = th.nn.functional.cross_entropy(
            bsouts, bslbls, reduction="none"
        )
        bsloss: th.Tensor = th.mean(bslosses)
        # update selector parameter
        opt.zero_grad()
        bsloss.backward()
        opt.step()
        # track metrics
        slosses_l.append(bslosses.detach().to(device="cpu"))
        bmetrics_d: dict[str, float] = {
            "bsloss": bsloss.item(),
        }
        pbar.set_postfix(bmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(bmetrics_d, "train"), step=tstate["opt_step"]
        )
        tstate["opt_step"] = tstate["opt_step"] + 1
    sloss_avg: th.Tensor = th.mean(th.cat(slosses_l, dim=0))
    metrics_d: dict[str, float] = {"sloss_avg": sloss_avg.item()}
    return metrics_d


def fit(
    tstate: _TrainState,
    xps: thd.TensorDict,
    init_fidx: int,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
):
    tloader = th_data.DataLoader(
        xps,
        batch_size=bsz,
        collate_fn=lambda bxp: _collate_fn(
            bxp, n_covs=tstate["selector"].n_covs, init_fidx=init_fidx
        ),
    )
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _ in pbar:
        metrics_d: dict[str, float] = _fit_iter(
            tstate=tstate, tloader=tloader, pbar=pbar, plf=plf
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "train"), step=tstate["n_fit_itr"]
        )
        tstate["n_fit_itr"] = tstate["n_fit_itr"] + 1
    pbar.close()


# %%
@th.no_grad()
def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    selector: SoftmaxSelector,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    selector.to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    # repeat feature acquisition until all features in template has been acquired.
    for _ in itrtls.count():
        # make feature bit mask
        _m: th.Tensor = th.zeros_like(x)
        _m[fobsd_l] = 1
        # forward prop. selector
        _sinps: th.Tensor = th.cat((x, _m))[None, :].to(device=plf.device)
        _souts: th.Tensor = selector(_sinps)
        # choose a template
        _fcomb_idx: int = int(th.argmax(_souts[0]).item())
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
    return pyhats[0], fobsd_l, fcomb


# %%
tcube, vcube = mydatasets.aaco.load_aaco_data("cube_20_0.3", to_normalize=False)
n_covs: int = tcube["xs"].shape[1]
n_labels: int = len(th.unique(tcube["ys"]))

# %%
allfcombs_l, init_fidx = make_templates(
    tcube, is_classification=True, max_features=5, top_k_features=10
)
fcomb_to_slbl: dict[tuple[int, ...], int] = {
    fcomb: i for i, fcomb in enumerate(allfcombs_l)
}
n_templates: int = len(allfcombs_l)

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

# %%
for _data in vcube:
    _pyhat, _, _ = run_one_random_episode(
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
xps = make_init_queries(
    tdata=tcube,
    classifier=classifier,
    init_fidx=init_fidx,
    allfcombs_l=allfcombs_l,
    fcomb_to_slbl=fcomb_to_slbl,
    init_capital=100,
    # init_capital=50 * len(tcube),
)

# %%
# plf = pl.Fabric()
plf = pl.Fabric(accelerator="cpu")

# %%
nnet = mymodels.nn.make_fcn(
    in_features=2 * tcube["xs"].shape[1],
    out_features=n_templates,
    layer_specs=[
        (tcube["xs"].shape[1], None, None, None),
        (tcube["xs"].shape[1], None, None, None),
    ],
)
selector = SoftmaxSelector(
    n_covs=n_covs,
    n_templates=n_templates,
    nnet=nnet,
)
opt = th.optim.Adam(selector.parameters())

# %%
tstate = _TrainState(selector=selector, opt=opt, n_trial_itr=0, n_fit_itr=0, opt_step=0)

# %%
fit(tstate=tstate, xps=xps, init_fidx=init_fidx, n_iter=1000, bsz=1024, plf=plf)


# %%
