from __future__ import annotations

from typing import TypedDict

import lightning as pl
import tensordict as thd
import torch as th


class _TrainState(TypedDict):
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
def _make_fit_bsinps_bstargs(
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


def _fit_iter_nnet_selector(
    tstate: _TrainState,
    bstdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    lmbda: float,
    plf: pl.Fabric,
) -> dict[str, float]:
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    # (bsz, 2 * n_covs) (bsz, )
    bsinps, bstargs = _make_fit_bsinps_bstargs(
        bstdata=bstdata, init_fidx=init_fidx, lmbda=lmbda, tmpls=tmpls
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
