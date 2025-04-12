from __future__ import annotations

from typing import TypedDict

import lightning as pl
import mymodels
import tafalib
import tensordict as thd
import torch as th
import torchrl.data as thrl_data


class _TrainState(TypedDict):
    rplbuf: thrl_data.ReplayBuffer
    nnet: th.nn.Module
    opt: th.optim.Optimizer
    fit_itr: int
    opt_step: int


@th.no_grad()
def compile_selector_dataset(
    tdata: thd.TensorDict, tpcomp: thd.TensorDict
) -> thd.TensorDict:
    # (n_data, n_covs)
    xs: th.Tensor = tdata["xs"]
    # (n_data, )
    ys: th.Tensor = tdata["ys"]
    # (n_data, n_tmpls, n_labels)
    pyhats: th.Tensor = tpcomp["pyhats"]
    # (n_data, n_tmpls)
    cels: th.Tensor = tpcomp["cels"]
    rwds: th.Tensor = tpcomp["rwds"]
    # # (n_data, n_tmpls)
    # slbls: th.Tensor = th.softmax(rwds / tau_rwd, dim=1)
    # bundle tensors into tensordict
    stdata = thd.TensorDict(
        {
            "xs": xs,
            "ys": ys,
            "pyhats": pyhats,
            "cels": cels,
            "rwds": rwds,
        }
    ).auto_batch_size_(1)
    return stdata


@th.no_grad()
def _make_warmup_fit_bsinps_bstargs(
    bstdata: thd.TensorDict, init_fidx: int, lmbda: float, tmpls: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    tmpls = tmpls.to(device=bxs.device)
    bfms: th.Tensor = tmpls[btmplidxs]
    n_covs: int = bxs.shape[1]
    # randomly drop features
    boms: th.Tensor = th.randint(0, 2, (bsz, n_covs), device=bxs.device)
    boms = th.clamp(bfms - boms, 0.0, 1.0)
    boms[:, init_fidx] = 1
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs * boms, boms), dim=1)
    # (bsz, n_tmpls, n_covs)
    bfms_avail: th.Tensor = th.clamp(tmpls[None, :, :] - boms[:, None, :], 0.0, 1.0)
    # (bsz, n_tmpls)
    bcels: th.Tensor = bstdata["cels"].to(device=bxs.device)
    bcosts: th.Tensor = bcels + lmbda * th.sum(bfms_avail, dim=2)
    # (bsz, )
    bstargs: th.Tensor = th.argmin(bcosts, dim=1)
    return bsinps, bstargs


def _warmup_fit_iter_nnet_selector(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    lmbda: float,
    plf: pl.Fabric,
) -> dict[str, float]:
    rplbuf: thrl_data.ReplayBuffer = tstate["rplbuf"]
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # (bsz, 2 * n_covs) (bsz, )
    bsinps, bstargs = _make_warmup_fit_bsinps_bstargs(
        bstdata=bstdata, init_fidx=init_fidx, lmbda=lmbda, tmpls=tmpls
    )
    rplbuf.extend(
        thd.make_tensordict(
            {"inps": bsinps.to(device="cpu"), "targs": bstargs.to(device="cpu")}
        ).auto_batch_size_(1)
    )
    # (bsz, n_tmpls)
    bsouts: th.Tensor = nnet(bsinps)
    # compute selector loss
    bslosses: th.Tensor = th.nn.functional.cross_entropy(
        bsouts, bstargs, reduction="none"
    )
    bsloss: th.Tensor = th.mean(bslosses)
    # update selector parameter
    opt.zero_grad()
    bsloss.backward()
    opt.step()
    # track metrics
    bmetrics_d: dict[str, float] = {
        "bsloss": bsloss.item(),
    }
    tstate["opt_step"] = tstate["opt_step"] + 1
    return bmetrics_d


@th.no_grad()
def _make_dagger_fit_bsinps_bstargs(
    bstdata: thd.TensorDict,
    bfobsds_l: list[list[int]],
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
) -> tuple[list[th.Tensor], list[th.Tensor]]:
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bcels: th.Tensor = bstdata["cels"]
    tmpls = tmpls.to(device=bxs.device)
    # make new masks
    boms_l: list[th.Tensor] = list()
    for _bidx, _fobsd_l in enumerate(bfobsds_l):
        assert _fobsd_l[0] == init_fidx
        _oms_l: list[th.Tensor] = list()
        for _i in range(len(_fobsd_l)):
            _om: th.Tensor = th.zeros_like(bxs[_bidx])
            _om[_fobsd_l[: _i + 1]] = 1
            _oms_l.append(_om)
        boms_l.append(th.stack(_oms_l))
    # make new selector inputs
    bsinps_l: list[th.Tensor] = list()
    bstargs_l: list[th.Tensor] = list()
    for _bidx, _om in enumerate(boms_l):
        _xs: th.Tensor = bxs[_bidx][None, :].expand(len(_om), -1)
        _sinps: th.Tensor = th.cat((_xs * _om, _om), dim=1)
        _fms_avail: th.Tensor = th.clamp(tmpls[None, :, :] - _om[:, None, :], 0.0, 1.0)
        _cels: th.Tensor = bcels[_bidx][None, :]
        _costs: th.Tensor = _cels + lmbda * th.sum(_fms_avail, dim=2)
        _stargs: th.Tensor = th.argmin(_costs, dim=1)
        bsinps_l.append(_sinps)
        bstargs_l.append(_stargs)
    return bsinps_l, bstargs_l


@th.no_grad()
def _update_dagger_replay_buffer_(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    plf: pl.Fabric,
):
    rplbuf: thrl_data.ReplayBuffer = tstate["rplbuf"]
    nnet: th.nn.Module = tstate["nnet"].eval().to(device=plf.device)
    _, _, _, bfobsds_l = tafalib.utils.predict(
        data=bstdata,
        classifier=classifier,
        cost_est=lambda x: tafalib.functional.selector_nnet_cost_est(
            x, nnet=nnet, device=plf.device
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    nnet.train()
    # shape of list (bsz, ) (bsz, )
    # shape of 1st tensor (len(boms[_bi]), 2 * n_covs)
    # shape of 2nd tensor (len(boms[_bi]), )
    bsinps_l, bstargs_l = _make_dagger_fit_bsinps_bstargs(
        bstdata=bstdata,
        bfobsds_l=bfobsds_l,
        init_fidx=init_fidx,
        lmbda=lmbda,
        tmpls=tmpls,
    )
    # (sum(map(len, bsinps_l)), 2 * n_covs)
    bsinps: th.Tensor = th.cat(bsinps_l, dim=0).to(device="cpu")
    # (sum(map(len, bstargs_l)), )
    bstargs: th.Tensor = th.cat(bstargs_l, dim=0).to(device="cpu")
    assert len(bsinps) == len(bstargs)
    # (sum(map(len, bsinps_l)))
    bdata = thd.make_tensordict({"inps": bsinps, "targs": bstargs}).auto_batch_size_(1)
    # add bdata to replay buffer
    rplbuf.extend(bdata)


def _dagger_fit_iter_nnet_selector(
    tstate: _TrainState, bsz: int, plf: pl.Fabric
) -> dict[str, float]:
    rplbuf: thrl_data.ReplayBuffer = tstate["rplbuf"]
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # sample experiences from replay buffer
    bdata = rplbuf.sample(batch_size=bsz)
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = bdata["inps"].to(device=plf.device)
    # (bsz, )
    bstargs: th.Tensor = bdata["targs"].to(device=plf.device)
    # (bsz, n_tmpls)
    bsouts: th.Tensor = nnet(bsinps)
    # compute selector loss
    # (bsz, )
    bslosses: th.Tensor = th.nn.functional.cross_entropy(
        bsouts, bstargs, reduction="none"
    )
    bsloss: th.Tensor = th.mean(bslosses)
    # update selector parameter
    opt.zero_grad()
    bsloss.backward()
    opt.step()
    # track metrics
    bmetrics_d: dict[str, float] = {
        "bsloss": bsloss.item(),
    }
    tstate["opt_step"] = tstate["opt_step"] + 1
    return bmetrics_d
