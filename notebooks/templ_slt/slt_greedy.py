# %%
from __future__ import annotations

import itertools as itrtls
from typing import TypedDict

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
class SoftmaxSelector(th.nn.Module):
    n_covs: int
    n_templates: int
    nnet: th.nn.Module

    def __init__(self, n_covs: int, n_tmpls: int, nnet: th.nn.Module) -> None:
        super().__init__()
        self.n_covs = n_covs
        self.n_templates = n_tmpls
        self.nnet = nnet

    def forward(self, sinps: th.Tensor, to_probs: bool) -> th.Tensor:
        outs: th.Tensor = self.nnet(sinps)
        if to_probs:
            outs = th.nn.functional.softmax(outs)
        return outs


# %%
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


@th.no_grad()
def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    selector: SoftmaxSelector,
    init_fidx: int,
    allfcombs_l: list[tuple[int, ...]],
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    selector.eval().to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    # repeat feature acquisition until all features in template has been acquired.
    for _ in itrtls.count():
        # make feature bit mask
        _m: th.Tensor = th.zeros_like(x)
        _m[fobsd_l] = 1
        # forward prop. selector
        _sinps: th.Tensor = th.cat((x, _m))[None, :].to(device=plf.device)
        _souts: th.Tensor = selector(_sinps, to_probs=False)
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
def make_templates(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls: int,
    n_cands: int,
    lmbda: float,
    max_features: int,
) -> th.Tensor:
    txs: th.Tensor = tdata["xs"]
    tys: th.Tensor = tdata["ys"]
    n_covs: int = txs.shape[1]
    tmpls: th.Tensor = th.zeros((n_tmpls, n_covs), dtype=th.long)
    tmpl_fcs: list[tuple[int, ...]] = list()
    for _i in tqdm.trange(
        n_tmpls, desc="make templates", leave=True, dynamic_ncols=True
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


@th.no_grad()
def compile_selector_dataset(
    tdata: thd.TensorDict,
    tmpls: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    lmbda: float,
) -> thd.TensorDict:
    classifier.eval()
    xs: th.Tensor = tdata["xs"]
    ys: th.Tensor = tdata["ys"]
    n_data: int = len(xs)
    n_tmpls: int = len(tmpls)
    # (n_data * n_tmpls, n_labels)
    pyhats_: th.Tensor = classifier.predict_proba(
        ctxs=xs[:, None, :].expand(-1, n_tmpls, -1).flatten(0, 1),
        acts=tmpls[None, :, :].expand(n_data, -1, -1).flatten(0, 1),
    )
    ys_: th.Tensor = ys[:, None].expand(n_data, n_tmpls).flatten(0, 1)
    # (n_data, n_tmpls)
    cels: th.Tensor = th.nn.functional.cross_entropy(
        pyhats_, ys_, reduction="none"
    ).unflatten(0, (n_data, n_tmpls))
    rwds: th.Tensor = -cels - lmbda * th.sum(tmpls, dim=1)[None, :]
    # (n_data)
    slbls: th.Tensor = th.argmax(rwds, dim=1)
    # bundle tensors into tensordict
    stdata = thd.TensorDict(
        {
            "xs": xs,
            "ys": ys,
            "cels": cels,
            "rwds": rwds,
            "slbls": slbls,
        }
    ).auto_batch_size_(1)
    return stdata


class _TrainState(TypedDict):
    selector: SoftmaxSelector
    opt: th.optim.Optimizer
    n_trial_itr: int
    n_fit_itr: int
    opt_step: int


@th.no_grad()
def _make_fit_bsinps(
    bstdata: thd.TensorDict, init_fidx: int, tmpls: th.Tensor
) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bfms: th.Tensor = tmpls[btmplidxs].to(device=bxs.device)
    # randomly drop features
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs))
    bnms[:, init_fidx] = 1
    bnms = th.cat((bnms, bnms), dim=1).to(device=bxs.device)
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs, bfms), dim=1)
    bsinps = bsinps * bnms
    return bsinps


def _fit_iter(
    tstate: _TrainState,
    tloader: th_data.DataLoader,
    init_fidx: int,
    tmpls: th.Tensor,
    pbar: tqdm.tqdm,
    plf: pl.Fabric,
) -> dict[str, float]:
    selector: SoftmaxSelector = tstate["selector"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    slosses_l: list[th.Tensor] = list()
    for bstdata in tloader:
        bstdata: thd.TensorDict
        bstdata = bstdata.to(device=plf.device)
        # (bsz, 2 * n_covs)
        bsinps: th.Tensor = _make_fit_bsinps(
            bstdata=bstdata, init_fidx=init_fidx, tmpls=tmpls
        )
        # (bsz, n_tmpls)
        bsouts: th.Tensor = selector(bsinps, to_probs=False)
        # compute selector loss
        bslosses: th.Tensor = th.nn.functional.cross_entropy(
            bsouts, th.softmax(bstdata["rwds"], dim=1), reduction="none"
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
    stdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
):
    tloader = th_data.DataLoader(
        stdata,  # type: ignore
        batch_size=bsz,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _ in pbar:
        metrics_d: dict[str, float] = _fit_iter(
            tstate=tstate,
            tloader=tloader,
            init_fidx=init_fidx,
            tmpls=tmpls,
            pbar=pbar,
            plf=plf,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "train"), step=tstate["n_fit_itr"]
        )
        tstate["n_fit_itr"] = tstate["n_fit_itr"] + 1
    pbar.close()


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
n_tmpls: int = 256
n_cands: int = 64
lmbda: float = 0.1
max_features: int = 5

# %%
tmpls: th.Tensor = make_templates(
    tdata=tcube,
    classifier=classifier,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls,
    n_cands=n_cands,
    lmbda=lmbda,
    max_features=max_features,
)
allfcombs_l: list[tuple[int, ...]] = [
    tuple(th.argwhere(_tmpl == 1).tolist()) for _tmpl in tmpls
]

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
stdata = compile_selector_dataset(
    tdata=tcube, tmpls=tmpls, classifier=classifier, lmbda=lmbda
)

# %%
plf = pl.Fabric()
# plf = pl.Fabric(accelerator="cpu")

# %%
nnet = mymodels.nn.make_fcn(
    in_features=2 * tcube["xs"].shape[1],
    out_features=n_tmpls,
    layer_specs=[
        (tcube["xs"].shape[1], None, None, None),
        (tcube["xs"].shape[1], None, None, None),
    ],
)
selector = SoftmaxSelector(
    n_covs=n_covs,
    n_tmpls=n_tmpls,
    nnet=nnet,
)
opt = th.optim.Adam(selector.parameters())

# %%
tstate = _TrainState(selector=selector, opt=opt, n_trial_itr=0, n_fit_itr=0, opt_step=0)

# %%
fit(
    tstate=tstate,
    stdata=stdata,
    init_fidx=init_fidx,
    tmpls=tmpls,
    n_iter=5000,
    bsz=4096,
    plf=plf,
)

# %%
for _data in vcube:
    _pyhat, _, _ = run_one_episode(
        x=_data["xs"],
        classifier=classifier,
        selector=selector,
        init_fidx=init_fidx,
        allfcombs_l=allfcombs_l,
        plf=plf,
    )
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
print(pd.Series(metrics_d))

# %%
