from __future__ import annotations

import itertools as itrtls
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
def run_one_episode_all_obsd(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
    student=None,
    training=False,
) -> tuple[th.Tensor, list[int], tuple[int, ...], th.Tensor, th.Tensor, list[float]]:
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

    state_vec_l: list[float] = []
    oracle_action_l: list[int] = []
    store_weight_l: list[float] = []
    for _ in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        if training:
            _costs: th.Tensor = cost_est(_inps)
            _tmpl_idx: int = int(th.argmin(_costs[0]).item())

            _fm_avail: th.Tensor = th.maximum(
                tmpls[_tmpl_idx] - _fm, th.as_tensor(0.0, device=plf.device)
            )

        # student action -> template index
        if student and student != {}:
            if not isinstance(student, dict):
                student_action = student.act(_inps[0, :].cpu(), training=training)

            else:
                # check if exists
                if f"{float(len(fobsd_l)) - 1}" not in student:
                    break

                student_action = student[f"{float(len(fobsd_l)) - 1}"].act(
                    _inps[0, :].cpu(), training=training
                )

            _fm_avail_student: th.Tensor = th.maximum(
                tmpls[int(student_action)] - _fm, th.as_tensor(0.0, device=plf.device)
            )

        if training:
            oracle_action_l.append(int(_tmpl_idx))
            state_vec_l.append(_inps.cpu().numpy().tolist())

            best_loss = th.min(_costs[0]).item()
            second_best_loss = th.min(_costs[0][_costs[0] != best_loss]).item()
            weight = second_best_loss - best_loss
            store_weight_l.append(weight)

        if student is None or student == {}:
            if th.sum(_fm_avail) == 0:
                break
            fobsd_l.append(int(th.argmax(_fm_avail).item()))
        else:
            if th.sum(_fm_avail_student) == 0:
                break
            fobsd_l.append(int(th.argmax(_fm_avail_student).item()))

    fcomb = tuple(sorted(fobsd_l))
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fobsd_l] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)

    state_vec: th.Tensor = th.tensor(state_vec_l)
    oracle_action: th.Tensor = th.tensor(oracle_action_l)
    return pyhats[0], fobsd_l, fcomb, state_vec, oracle_action, store_weight_l


def evaluate(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
    student=None,
    training=False,
    buffer_state=None,
    buffer_action=None,
    buffer_weight=None,
    lmbda: float = 0.0,
):
    """compute metrics through rolling out policy over a dataset

    Args:
        data (thd.TensorDict): (n, ) dataset of interest; must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): cost estimator
        init_fidx (int): initial feature index
        tmpls (th.Tensor): (n_tmpls, n_covs) collection of templates
        metrics_func (thm.MetricCollection): a collection of metrics of interests
        plf (pl.Fabric): plf instance for prediction

    Returns:
        dict[str, float]: a dictionary of metrics
    """
    buffer_state = buffer_state if buffer_state is not None else list()
    buffer_action = buffer_action if buffer_action is not None else list()
    buffer_weight = buffer_weight if buffer_weight is not None else list()

    snfobsd_l: list[int] = list()
    snfcomb_l: list[int] = list()
    metrics_func.reset()

    all_ce_losses = []

    for _data in data:
        _pyhat, _fobsd_l, _fcomb, _state_vec, _oracle_action, _store_weight = (
            run_one_episode_all_obsd(
                x=_data["xs"],
                classifier=classifier,
                cost_est=cost_est,
                init_fidx=init_fidx,
                tmpls=tmpls,
                plf=plf,
                student=student,
                training=training,
            )
        )

        if len(_state_vec) != 0 and training:
            # single tree
            if isinstance(buffer_state, list):
                buffer_state.append(_state_vec)
                buffer_action.append(_oracle_action)
            else:
                # multiple trees
                for idx, state in enumerate(_state_vec):
                    buffer_state[f"{float(idx)}"].append(state)
                    buffer_action[f"{float(idx)}"].append(_oracle_action[idx])
                    buffer_weight[f"{float(idx)}"].append(_store_weight[idx])

        snfobsd_l.append(len(_fobsd_l))
        snfcomb_l.append(len(_fcomb))
        metrics_func.update(
            _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
        )

        # compute CE
        logits = _pyhat
        target = _data["ys"].to(logits.device)
        log_p = th.log(logits + 1e-12)
        ce = th.nn.functional.nll_loss(log_p, target)

        all_ce_losses.append(ce.item())

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

    avg_ce = sum(all_ce_losses) / len(all_ce_losses)
    obj = avg_ce + th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)) * lmbda

    if training:
        return metrics_d, buffer_state, buffer_action, buffer_weight

    else:
        return metrics_d, None, None, obj
