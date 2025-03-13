from __future__ import annotations

import itertools as itrtls
from typing import Callable

import lightning as pl
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import tensordict as thd
import torch as th
import torch.distributions.utils
import torchmetrics as thm
import tqdm.auto as tqdm


def feature_masks_to_feature_combs(tmpls: th.Tensor):
    return [tuple(th.argwhere(_tmpl == 1).flatten().tolist()) for _tmpl in tmpls]


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
    tmpls = tmpls.to(plf.device)
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
        _costs: th.Tensor = cost_est(_inps.to(device=plf.device))
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


def evaluate(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
) -> dict[str, float]:
    snfobsd_l: list[int] = list()
    snfcomb_l: list[int] = list()
    metrics_func.reset()
    for _data in data:
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
        metrics_func.update(
            _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
        )
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    metrics_d.update(
        {
            "init_fidx": init_fidx,
            "feature observed": th.mean(
                th.as_tensor(snfobsd_l, dtype=th.float32)
            ).item(),
            "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
        }
    )
    return metrics_d
