from __future__ import annotations

import itertools as itrtls
import time
from typing import Callable

import lightning as pl
import mymodels
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm


def feature_masks_to_feature_combs(tmpls: th.Tensor):
    return [tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls]


@th.no_grad()
def precomp_rwds_for_tmpls(
    tmpls: th.Tensor,
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> thd.TensorDict:
    """precompute rewards rewards for using templates over passed in dataset.

    Args:
        tmpls (th.Tensor): (n_tmpls, n_covs) set of templates
        data (thd.TensorDict): (n, ) dataset of interest; must contain key `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature classifier
        lmbda (float): coefficient for feature cost
        bsz (int): batch size
        plf (pl.Fabric): lightning fabric instance

    Returns::
        thd.TensorDict: a TensorDict have same length as data containing following keys
            `pyhats`: predicted class probability (n, n_tmpls, n_labels)
            `cels`: (n, n_tmpls) cross entropy loss of each instance
            `rwds`: (n, n_tmpls) rewards of each instance
    """
    classifier.eval().to(device=plf.device)
    txs: th.Tensor = data["xs"]
    tys: th.Tensor = data["ys"]
    n_cands: int = len(tmpls)
    n_labels: int = len(th.unique(tys))
    # (n_data,  n_cands, n_labels)
    pyhats: th.Tensor = th.empty((len(txs), n_cands, n_labels), dtype=th.float32)
    # (n_data,  n_cands)
    cels: th.Tensor = th.empty((len(txs), n_cands), dtype=th.float32)
    rwds: th.Tensor = th.empty_like(cels)
    # loop over dataset using flattened indices tensor splitted with bsz
    pbar = tqdm.tqdm(
        th.split(th.cartesian_prod(th.arange(len(data)), th.arange(n_cands)), bsz),
        desc="precomp candidates",
        leave=False,
        dynamic_ncols=True,
    )
    for _bidxs in pbar:
        # (_bsz, n_covs)
        _bacts: th.Tensor = tmpls[_bidxs[:, 1], :].to(device=plf.device)
        _bctxs: th.Tensor = txs[_bidxs[:, 0], :].to(device=plf.device) * _bacts
        # (_bsz, n_labels)
        _bpyhats: th.Tensor = classifier.predict_proba(_bctxs, _bacts)
        # (_bsz, )
        _bcels: th.Tensor = th.nn.functional.nll_loss(
            th.log(_bpyhats), tys[_bidxs[:, 0]].to(device=plf.device), reduction="none"
        )
        _brwds: th.Tensor = -_bcels - lmbda * th.sum(_bacts, dim=1)
        # set it back to result tensor
        pyhats[_bidxs[:, 0], _bidxs[:, 1], :] = _bpyhats.to(device="cpu")
        cels[_bidxs[:, 0], _bidxs[:, 1]] = _bcels.to(device="cpu")
        rwds[_bidxs[:, 0], _bidxs[:, 1]] = _brwds.to(device="cpu")
    pbar.close()
    # turn into tensordict
    tpcomp = thd.TensorDict(
        {"pyhats": pyhats, "cels": cels, "rwds": rwds}
    ).auto_batch_size_(1)
    return tpcomp


@th.no_grad()
def run_one_episode(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    """run one episode using **only** features from the last used template

    Args:
        x (th.Tensor): (n_covs, ) instance of interest
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): a cost estimator mapping a tensor of shape (n, 2 * n_covs) to a tensor of (n, n_tmpls)
        init_fidx (int): initial feature indices
        tmpls (th.Tensor): (n_tmpls, n_covs)
        plf (pl.Fabric): lightning fabric instance

    Returns::
        th.Tensor: (n_labels, ) predicted class probability
        list[int]: (n_feats_obsd, ) a list of feature indices in the order of acquisition.
        tuple[int, ...]]: (n_feats_obsd, ) a tuple of feature indices acquired; does NOT preseve acquisition order
    """
    classifier.eval().to(device=plf.device)
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    x = x.to(device=plf.device)
    tmpls = tmpls.to(device=plf.device)
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
        _costs: th.Tensor = cost_est(_inps)
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(
            tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0, device=plf.device)
        )
        if th.sum(_fm_avail) == 0:
            fcomb = tuple(th.argwhere(tmpls[_tmpl_idx] == 1).flatten().tolist())
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    assert fcomb is not None
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fcomb] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


@th.no_grad()
def run_one_episode_all_obsd(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    """run one episode using **all** features acquired

    Args:
        x (th.Tensor): (n_covs, ) instance of interest
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): a cost estimator mapping a tensor of shape (n, 2 * n_covs) to a tensor of (n, n_tmpls)
        init_fidx (int): initial feature indices
        tmpls (th.Tensor): (n_tmpls, n_covs)
        plf (pl.Fabric): lightning fabric instance

    Returns::
        th.Tensor: (n_labels, ) predicted class probability
        list[int]: (n_feats_obsd, ) a list of feature indices in the order of acquisition.
        tuple[int, ...]]: (n_feats_obsd, ) a tuple of feature indices acquired; does NOT preseve acquisition order
    """
    classifier.eval().to(device=plf.device)
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    x = x.to(device=plf.device)
    tmpls = tmpls.to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        _costs: th.Tensor = cost_est(_inps)
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(
            tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0, device=plf.device)
        )
        if th.sum(_fm_avail) == 0:
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    fcomb = tuple(sorted(fobsd_l))
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fobsd_l] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


def evaluate(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    lmbda: float,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
) -> dict[str, float]:
    """compute metrics through rolling out policy over a dataset

    Args:
        data (thd.TensorDict): (n, ) dataset of interest; must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): cost estimator
        init_fidx (int): initial feature index
        tmpls (th.Tensor): (n_tmpls, n_covs) collection of templates
        lmbda (float): penalty for acquiring more features
        metrics_func (thm.MetricCollection): a collection of metrics of interests
        plf (pl.Fabric): plf instance for prediction

    Returns:
        dict[str, float]: a dictionary of metrics
    """
    snfobsd_l: list[int] = list()
    snfcomb_l: list[int] = list()
    pyhats_l: list[th.Tensor] = list()
    fms: th.Tensor = th.zeros((len(data), data["xs"].shape[1]), dtype=th.long)
    metrics_func.reset()
    start_time_ns: int = time.time_ns()
    for _i, _data in tqdm.tqdm(
        enumerate(data), desc="eval", total=len(data), dynamic_ncols=True, leave=False
    ):
        _pyhat, _fobsd_l, _fcomb = run_one_episode_all_obsd(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        snfobsd_l.append(len(_fobsd_l))
        snfcomb_l.append(len(_fcomb))
        pyhats_l.append(_pyhat[None, :].to(device="cpu"))
        fms[_i, _fcomb] = 1
        metrics_func.update(
            _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
        )
    end_time_ns: int = time.time_ns()
    pyhats: th.Tensor = th.cat(pyhats_l, dim=0)
    cels: th.Tensor = th.nn.functional.cross_entropy(
        pyhats, data["ys"], reduction="none"
    )
    rwds: th.Tensor = -cels - lmbda * th.sum(fms, dim=1)
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    metrics_d.update(
        {
            "init_fidx": init_fidx,
            "reward": th.mean(rwds, dim=0).item(),
            "feature observed": th.mean(
                th.as_tensor(snfobsd_l, dtype=th.float32)
            ).item(),
            "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
            "inference_time_ns": end_time_ns - start_time_ns,
            "avg_pred_time_ns": (end_time_ns - start_time_ns) / len(data),
        }
    )
    return metrics_d


def predict(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, list[list[int]]]:
    """make predictions through rolling out policy over a dataset

    Args:
        data (thd.TensorDict): (n, ) dataset of interest; must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): cost estimator
        init_fidx (int): initial feature index
        tmpls (th.Tensor): (n_tmpls, n_covs) collection of templates
        metrics_func (thm.MetricCollection): a collection of metrics of interests
        plf (pl.Fabric): plf instance for prediction

    Returns::
        th.Tensor: (n, n_labels) predicted class probability
        th.Tensor: (n, n_covs) last template selected
        th.Tensor: (n, n_covs) indicators for feature observed
        list[tuple[int, ...]]]: a list of length n with each element being a length (n_feats_obsd, ) tuple of feature indices acquired; does NOT preseve acquisition order
    """
    n_labels: int = classifier.n_labels
    pyhats: th.Tensor = th.empty((len(data), n_labels), dtype=th.float32)
    oms: th.Tensor = th.zeros_like(data["xs"])
    fms: th.Tensor = th.zeros_like(data["xs"])
    fobsds_l: list[list[int]] = list()
    for _i, _data in enumerate(data):
        _pyhat, _fobsd_l, _fcomb = run_one_episode(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        pyhats[_i] = _pyhat
        oms[_i, _fcomb] = 1
        fms[_i, _fobsd_l] = 1
        fobsds_l.append(_fobsd_l)
    return pyhats, oms, fms, fobsds_l


@th.no_grad()
def run_one_episode_all_obsd_with_missing_feature(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    missing_value: float,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    """run one episode using **all** features acquired

    Args:
        x (th.Tensor): (n_covs, ) instance of interest
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): a cost estimator mapping a tensor of shape (n, 2 * n_covs) to a tensor of (n, n_tmpls)
        init_fidx (int): initial feature indices
        tmpls (th.Tensor): (n_tmpls, n_covs)
        plf (pl.Fabric): lightning fabric instance

    Returns::
        th.Tensor: (n_labels, ) predicted class probability
        list[int]: (n_feats_obsd, ) a list of feature indices in the order of acquisition.
        tuple[int, ...]]: (n_feats_obsd, ) a tuple of feature indices acquired; does NOT preseve acquisition order
    """
    classifier.eval().to(device=plf.device)
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    x = x.to(device=plf.device)
    # handle the case where certain feature is completely missing in given dataset
    tmpls = th.where(
        (x == missing_value)[None, :].expand((len(tmpls), -1)), 0, tmpls
    ).to(device=plf.device)
    # in case initial feature is not observed
    if th.any(tmpls[:, init_fidx]) == 0:
        _feature_avail: list[int] = (
            th.argwhere(th.sum(tmpls, dim=0) > 0).flatten().tolist()
        )
        _ridx: int = int(th.randint(0, len(_feature_avail), ()).item())
        init_fidx = _feature_avail[_ridx]
    # rollout current instance
    fobsd_l: list[int] = [init_fidx]
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        _costs: th.Tensor = cost_est(_inps)
        _tmpl_idx: int = int(th.argmin(_costs[0]).item())
        _fm_avail: th.Tensor = th.maximum(
            tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0, device=plf.device)
        )
        if th.sum(_fm_avail) == 0:
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    fcomb = tuple(sorted(fobsd_l))
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fobsd_l] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


def evaluate_with_missing_feature(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    missing_value: float,
    lmbda: float,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
) -> dict[str, float]:
    """compute metrics through rolling out policy over a dataset

    Args:
        data (thd.TensorDict): (n, ) dataset of interest; must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): cost estimator
        init_fidx (int): initial feature index
        tmpls (th.Tensor): (n_tmpls, n_covs) collection of templates
        lmbda (float): penalty for acquiring more features
        metrics_func (thm.MetricCollection): a collection of metrics of interests
        plf (pl.Fabric): plf instance for prediction

    Returns:
        dict[str, float]: a dictionary of metrics
    """
    snfobsd_l: list[int] = list()
    snfcomb_l: list[int] = list()
    pyhats_l: list[th.Tensor] = list()
    fms: th.Tensor = th.zeros((len(data), data["xs"].shape[1]), dtype=th.long)
    metrics_func.reset()
    start_time_ns: int = time.time_ns()
    for _i, _data in tqdm.tqdm(
        enumerate(data), desc="eval", total=len(data), dynamic_ncols=True, leave=False
    ):
        _pyhat, _fobsd_l, _fcomb = run_one_episode_all_obsd_with_missing_feature(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            missing_value=missing_value,
            plf=plf,
        )
        snfobsd_l.append(len(_fobsd_l))
        snfcomb_l.append(len(_fcomb))
        pyhats_l.append(_pyhat[None, :].to(device="cpu"))
        fms[_i, _fcomb] = 1
        metrics_func.update(
            _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
        )
    end_time_ns: int = time.time_ns()
    pyhats: th.Tensor = th.cat(pyhats_l, dim=0)
    cels: th.Tensor = th.nn.functional.cross_entropy(
        pyhats, data["ys"], reduction="none"
    )
    rwds: th.Tensor = -cels - lmbda * th.sum(fms, dim=1)
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    metrics_d.update(
        {
            "init_fidx": init_fidx,
            "reward": th.mean(rwds, dim=0).item(),
            "feature observed": th.mean(
                th.as_tensor(snfobsd_l, dtype=th.float32)
            ).item(),
            "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
            "inference_time_ns": end_time_ns - start_time_ns,
            "avg_pred_time_ns": (end_time_ns - start_time_ns) / len(data),
        }
    )
    return metrics_d
