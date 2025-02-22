# %%
from __future__ import annotations

import itertools as itrtls
import math
import os
from typing import Callable, Optional

import numpy as np

import _classifiers
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import pandas as pd
import tensordict as thd
import torch as th
import torch.distributions.utils
import torchmetrics as thm
import tqdm.auto as tqdm


# %%
def feature_masks_to_feature_combs(tmpls: th.Tensor):
    return [tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls]


# %%
def eval_masks(
    X,
    B,
    classifiers: mymodels.classifiers.SubsetFeatureClassifier,
    Y=None,
    bsize=16,
    matched_B=False,
    classifier_filters=None,
):
    """
    Evaluate an ensemble of classifiers on tuples of inputs and masks.
    Args:
      X: N x d data matrix
      B: M x d mask matrix
      Y: N array of labels. If Y is given then the negative log likelihoods
        is returned else predictions are returned.
      If matched_B is True then N==M and instances are only evaluated on
        respective masks; else (default), all instances are evaluated on each
        mask in B.
      classifier_filter: N x nests matrix of indicators of which classifiers to
        use for instances
    Returns: N x M nlls if Y is given else N x M x nclasses predictions (when
      matched_B=False).
    """
    N = X.shape[0]
    M = B.shape[0]

    if Y is not None:
        if matched_B:
            out = np.zeros((N,), np.float32)
        else:
            out = np.zeros((N, M), np.float32)
    else:
        if matched_B:
            out = np.zeros((N, classifiers.n_labels), np.float32)
        else:
            out = np.zeros((N, M, classifiers.n_labels), np.float32)
    for bi in range(0, N, bsize):
        Xbatch = X[bi : bi + bsize]
        Ybatch = Y[bi : bi + bsize]
        if matched_B:
            Bbatch = B[bi : bi + bsize]
        else:
            Bbatch = np.tile(B, (Xbatch.shape[0], 1))
            Xbatch = np.repeat(Xbatch, M, 0)

        # Xbatch = np.concatenate((Xbatch * Bbatch, Bbatch), 1)
        preds: np.ndarray = classifiers.predict_proba(
            th.as_tensor(Xbatch * Bbatch, dtype=th.float32),
            th.as_tensor(Bbatch, dtype=th.float32),
        ).numpy()
        # preds = np.stack([est.predict_proba(Xbatch) for est in classifiers], -1)
        # if classifier_filters is None:
        #     preds = np.mean(preds, -1)
        # else:
        #     split_batch = classifier_filters[bi : bi + bsize]
        #     split_batch = np.repeat(split_batch, M, 0)
        #     nests = np.sum(split_batch, -1, keepdims=True)
        #     # print(preds.shape)
        #     # print(split_batch.shape)
        #     # print(nests.shape)
        #     preds = np.sum(preds * split_batch[:, None, :], -1) / nests
        if not matched_B:
            preds = preds.reshape(-1, M, classifiers.n_labels)
        if Y is None:
            out[bi : bi + bsize, ...] = preds
        else:
            out[bi : bi + bsize, ...] = np.stack(
                [-np.log(preds[i, ..., y]) for i, y in enumerate(Ybatch)], 0
            )
    return out


def get_mask_losses(
    Xtrn, Ytrn, B, classifiers, featcost, bsize=256, classifier_filters=None
):
    # TODO: use filters?
    mask_costs = featcost * np.sum(B, 1)

    bsize = 256
    Xcosts = []
    for si in range(0, Xtrn.shape[0], bsize):
        print(si)
        if classifier_filters is not None:
            nlls = eval_masks(
                Xtrn[si : si + bsize],
                B,
                classifiers,
                Y=Ytrn[si : si + bsize],
                classifier_filters=classifier_filters[si : si + bsize],
            )
        else:
            nlls = eval_masks(
                Xtrn[si : si + bsize], B, classifiers, Y=Ytrn[si : si + bsize]
            )
        Xcosts.append(nlls + mask_costs)

    return np.concatenate(Xcosts, 0)


def tf_arbitrary_knn(Xtrn, Ytrn, Xtst, indices, k=1, Xtrnl2=None, toss_first=False):
    """
    Args:
      Xtrn: N x d Train Instances
      Ytrn: N x nclass Train Labels (e.g., one-hot)
      Xtst: M x d Query Instances
      indices: list of features to use
      k: number of neighbors
      Xtrnl2: N x 1 vector of squared norms of Xtrn instances
      toss_first: flag to throwout the first neighbor
                  (if querying within training set)
    """

    # Xtrnfeats = np.take_along_axis(Xtrn, indices, axis=1)
    # Xtstfeats = np.take_along_axis(Xtst, indices, axis=1)
    Xtrnfeats = Xtrn[:, indices]
    Xtstfeats = Xtst[:, indices]
    if Xtrnl2 is None:
        Xtrnl2 = np.sum(Xtrnfeats**2, axis=1, keepdims=True)
    d2 = (
        Xtrnl2
        - 2.0 * np.matmul(Xtrnfeats, Xtstfeats.T)
        + np.transpose(np.sum(Xtstfeats**2, axis=1, keepdims=True))
    )
    d2_sorti = np.argsort(d2, axis=0)
    print(d2.shape)
    print(d2_sorti.flatten()[:k])
    ntmpl: int = Ytrn.shape[1]
    n: int = d2_sorti.shape[1]
    d2_sorti = th.as_tensor(d2_sorti)[:, None, :].expand(-1, ntmpl, -1).numpy()

    ys = th.as_tensor(Ytrn)[:, :, None].expand(-1, -1, n).numpy()
    Y_neighbors = np.mean(
        np.take_along_axis(
            ys,
            d2_sorti[(1 if toss_first else 0) : (k + 1 if toss_first else k), :, :],
            axis=0,
        ),
        axis=0,
    ).T
    return Y_neighbors


def eval_rollout(
    Xval,
    Yval,
    startdim,
    featcost,
    Xtrn,
    Ytrn,
    tempNLLtrn,
    B_temp,
    classifiers: mymodels.classifiers.SubsetFeatureClassifier,
    metric_func: thm.MetricCollection,
    ao_k=5,
):
    d = Xval.shape[1]
    eyed = np.eye(d, dtype=np.float32)
    eyedplusone = np.eye(d + 1, dtype=np.float32)

    accu_ao = []
    bs_ao = []
    # ao_k = 5
    metric_func.reset()
    for i in range(Xval.shape[0]):
        print("\n\n\n{}".format(i))
        # i = np.random.randint(Xval.shape[0])
        for s in range(d + 1):
            if s == 0:
                # if in the first step, draw action from null_distribution
                # b_curr = np.float32(np.random.multinomial(1, p_null))
                b_curr = eyed[startdim, None, :]  # Deterministic start
                print(b_curr)
                continue  # move to next step
            elif s < d:
                xo = Xval[i, None, :] * b_curr

                # nllroll = tf_arbitrary_knn(
                #     Xtrn, tempNLLtrn, xo, np.flatnonzero(b_curr[0]), k=ao_k
                # )

                mask_needed = np.maximum(
                    B_temp - b_curr, 0.0
                )  # fine if we have additional feats
                # mask_costs = featcost * np.sum(mask_needed, 1)

                # classifier_costs = nllroll + mask_costs
                classifier_costs = knn_cost_est(
                    th.cat((th.as_tensor(xo), th.tensor(b_curr)), dim=1),
                    lmbda=featcost,
                    txs=th.as_tensor(Xtrn),
                    tcels=th.as_tensor(tempNLLtrn),
                    tmpls=th.as_tensor(B_temp),
                    n_neighs=ao_k,
                    p=2,
                ).numpy()
                best_clss = np.argmin(classifier_costs)
                # print('>>{}'.format(np.sort(classifier_costs)))
                # print('>>{}'.format(mask_needed[best_clss, :]))
                a_pred = np.concatenate(
                    [mask_needed[best_clss, :], [0.0]]
                )  # TODO: tie break with softmax combination?
                # a_pred[:-1] = a_pred[:-1] - 1e8*b_curr[0]  # Don't acquire what's already acquired
                a_pred[:-1] = a_pred[:-1] * (1.0 - b_curr[0])
                if np.sum(a_pred[:-1]) == 0.0:
                    a_pred[-1] = 1.0
                else:
                    a_pred[:-1] = a_pred[:-1] / np.sum(a_pred[:-1])
                    a_pred[:-1] = a_pred[:-1] * (1.0 - a_pred[-1])
                action = eyedplusone[:, np.argmax(a_pred)]  # TODO: case with ties?
                # a_pred = le.inverse_transform(
                #   policy.predict_proba(np.concatenate((xo, b_curr), -1))[0, :])
                # action = eyedplusone[:, np.argmax(a_pred)]
                if action[-1] == 0.0:  # not predicting
                    b_curr = b_curr + action[:-1]
                    print(b_curr)
                    continue  # move to next step
            # make prediction
            print(np.flatnonzero(b_curr[0]))
            y_pred = classifiers.predict_proba(
                th.as_tensor(Xval[i : i + 1]), th.as_tensor(b_curr)
            ).numpy()
            # if bank is not None:
            #     est = bank.get_estimator(b_curr[0])
            #     y_pred = est.predict_proba(Xval[i : i + 1, b_curr[0] > 0])
            # elif classifiers is not None:
            #     y_pred = np.mean(
            #         np.stack(
            #             [
            #                 est.predict_proba(np.concatenate((xo, b_curr), 1))
            #                 for est in classifiers
            #             ],
            #             -1,
            #         ),
            #         -1,
            #     )
            # else:
            #     est = XGBClassifier(n_estimators=40)
            #     est.fit(Xtrn[:, b_curr[0] > 0], Ytrn)
            #     y_pred = est.predict_proba(Xval[i : i + 1, b_curr[0] > 0])
            metric_func.update(
                th.as_tensor(y_pred, dtype=th.float32), th.as_tensor(Yval[i][None])
            )
            print(y_pred, np.argmax(y_pred[0, :]), int(Yval[i]))
            accu_ao.append(np.argmax(y_pred[0, :]) == int(Yval[i]))
            print("correct!") if accu_ao[-1] else print("wrong!")
            bs_ao.append(b_curr)
            print(np.mean(accu_ao))
            print(np.mean([np.sum(b) for b in bs_ao]))
            break
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    print(pd.Series(metrics_d))
    return np.mean(accu_ao), np.mean([np.sum(b) for b in bs_ao])


# %%
def precomp_rwds_for_tmpls(
    tmpls: th.Tensor,
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    lmbda: float,
    bsz: int,
) -> thd.TensorDict:
    txs: th.Tensor = data["xs"]
    tys: th.Tensor = data["ys"]
    n_cands: int = len(tmpls)
    n_labels: int = len(th.unique(tys))
    # (n_data,  n_cands, n_labels)
    pyhats: th.Tensor = th.empty((len(txs), n_cands, n_labels), dtype=th.float32)
    # (n_data,  n_cands)
    cels: th.Tensor = th.empty((len(txs), n_cands), dtype=th.float32)
    rwds: th.Tensor = th.empty_like(cels)
    pbar = tqdm.tqdm(
        th.split(th.arange(0, len(txs), dtype=th.long), bsz),
        desc="precomp candidates",
        leave=False,
        dynamic_ncols=True,
    )
    for _btidxs in pbar:
        _bsz: int = len(_btidxs)
        # (_bsz * n_cands, n_covs)
        _bctxs: th.Tensor = txs[_btidxs, None, :].expand(-1, n_cands, -1).flatten(0, 1)
        _bacts: th.Tensor = tmpls[None, :, :].expand(_bsz, -1, -1).flatten(0, 1)
        _bctxs = _bctxs * _bacts
        # (_bsz * n_cands, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(_bctxs, _bacts)
        _blyhats: th.Tensor = torch.distributions.utils.probs_to_logits(_bpyhats)
        # (_bsz * n_cands)
        _btys: th.Tensor = tys[_btidxs, None].expand(-1, n_cands).flatten(0, 1)
        _bcels: th.Tensor = th.nn.functional.cross_entropy(
            _blyhats, _btys, reduction="none"
        )
        _brwds: th.Tensor = -_bcels - lmbda * th.sum(_bacts, dim=1)
        pyhats[_btidxs] = _bpyhats.unflatten(0, (_bsz, n_cands))
        cels[_btidxs] = _bcels.unflatten(0, (_bsz, n_cands))
        rwds[_btidxs] = _brwds.unflatten(0, (_bsz, n_cands))
    pbar.close()
    # turn into tensordict
    tpcomp = thd.TensorDict(
        {"pyhats": pyhats, "cels": cels, "rwds": rwds}
    ).auto_batch_size_(1)
    return tpcomp


@th.no_grad()
def _knn(
    xs: th.Tensor, txs: th.Tensor, n_neighs: int, p: float = 2
) -> tuple[th.Tensor, th.Tensor]:
    """knn

    Args:
        xs (th.Tensor): (n, n_covs) inputs
        txs (th.Tensor): (n_tdata, n_covs) training input covariates
        n_neighs (int): number of neighbors
        p (float, optional): p-norm distance. Defaults to 2.

    Returns::
        th.Tensor: (n, n_neighs) distance to nearest neighbors neighbors
        th.Tensor: (n, n_neighs) indices to nearest neighbors in training data
    """
    # (n, n_tdata)
    ds: th.Tensor = th.cdist(xs[None, :, :], txs[None, :, :], p=p)[0]
    # (n, n_tdata)
    ds, knnidxs = th.sort(ds, dim=1, descending=False)
    ds = ds[:, :n_neighs]
    knnidxs = knnidxs[:, :n_neighs]
    return ds, knnidxs


def knn_cost_est(
    inps: th.Tensor,
    lmbda: float,
    txs: th.Tensor,
    tcels: th.Tensor,
    tmpls: th.Tensor,
    n_neighs: int,
    p: float = 2,
) -> th.Tensor:
    """use knn strategy to compute cost

    Args:
        inps (th.Tensor): (n, n_covs * 2)
        lmbda (float): feature penalty
        txs (th.Tensor): (n, n_covs)
        tcels (th.Tensor): (n, n_tmpls)
        tmpls (th.Tensor): (n_tmpls, n_covs)
        n_neighs (int): number of neighbors
        p (float, optional): p-norm distance. Defaults to 2.

    Returns::
        th.Tensor: (n, n_tmpls) costs of using each template
    """
    # th.Tensor: (n, n_tmpls) costs of using each template
    n_covs: int = txs.shape[1]
    device: th.device = inps.device
    txs = txs.to(device=device)
    tcels = tcels.to(device=device)
    costs_l: list[th.Tensor] = list()
    for _inp in inps:
        # (n_covs, )
        _inp: th.Tensor
        _fm: th.Tensor = _inp[n_covs:]
        _fc: tuple[int, ...] = tuple(th.argwhere(_fm == 1).flatten().tolist())
        # (n_tmpls, n_covs)
        _fm_avail: th.Tensor = th.maximum(tmpls - _fm[None, :], th.as_tensor(0.0))
        # (n_neighs, )
        _knnidxs: th.Tensor = _knn(
            xs=_inp[None, _fc].to(device=device),
            txs=txs[:, _fc].to(device=device),
            n_neighs=n_neighs,
            p=p,
        )[1][0].to(device="cpu")
        # (n_tmpls, )
        _cels: th.Tensor = th.mean(tcels[_knnidxs], dim=0)
        _costs: th.Tensor = _cels + lmbda * th.sum(_fm_avail, dim=1)
        costs_l.append(_costs)
    costs: th.Tensor = th.stack(costs_l, dim=0).to(device=device)
    return costs


def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    fcomb: tuple[int, ...] | None = None
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        _costs: th.Tensor = cost_est(_inps.to(device=plf.device)).to(device="cpu")
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0))
        if th.sum(_fm_avail) == 0:
            fcomb = tuple(th.argwhere(tmpls[_tmpl_idx] == 1).flatten().tolist())
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    assert fcomb is not None
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fcomb] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


def run_one_episode_all_obsd(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        _costs: th.Tensor = cost_est(_inps.to(device=plf.device)).to(device="cpu")
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0))
        if th.sum(_fm_avail) == 0:
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    fcomb = tuple(sorted(fobsd_l))
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fobsd_l] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


# %%
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))

# %%
plf = pl.Fabric(accelerator="cpu")
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)
tclassifier = _classifiers.SubsetFeatureConcatXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgb_kwargs={"n_estimators": 40},
    fraction_training_data_per_split=1.0,
    n_splits=64,
    n_tmpl_per_instance=4,
)
vclassifier = mymodels.classifiers.SubsetFeatureXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgbc_kwargs={"n_estimators": 40},
)

# %%
init_fidx: int = 35
tmpls: th.Tensor = th.load("tmpls.pt")
lmbda: float = 0.075

# %%
tclassifier.fit_(tmpls)

# %%
tpcomp: thd.TensorDict = precomp_rwds_for_tmpls(
    tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=0.0, bsz=256
)

# %%
o1, o2 = eval_rollout(
    vdata["xs"].numpy(),
    vdata["ys"].numpy(),
    init_fidx,
    lmbda,
    tdata["xs"].numpy(),
    tdata["ys"].numpy(),
    tpcomp["cels"].numpy(),
    tmpls.numpy(),
    classifiers=vclassifier,
    metric_func=metrics_func,
)

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=5,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    snfobsd_l.append(len(_fobsd_l))
    snfcomb_l.append(len(_fcomb))
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "init_fidx": init_fidx,
        "feature used & observed": th.mean(
            th.as_tensor(snfobsd_l, dtype=th.float32)
        ).item(),
    }
)
print(pd.Series(metrics_d))

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = run_one_episode(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=5,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    snfobsd_l.append(len(_fobsd_l))
    snfcomb_l.append(len(_fcomb))
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "init_fidx": init_fidx,
        "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
        "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
    }
)
print(pd.Series(metrics_d))

# %%
