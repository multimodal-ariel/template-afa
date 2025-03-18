from __future__ import annotations

import mylib.ml
import torch as th


@th.no_grad()
def knn_cost_est(
    inps: th.Tensor,
    lmbda: float,
    txs: th.Tensor,
    tcels: th.Tensor,
    tmpls: th.Tensor,
    n_neighs: int,
    p: float = 2,
    is_train: bool = False,
    device: th.device = th.device("cpu"),
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
        is_train (bool, optional): whether the given `inps` is from training set. Defaults to `False`.

    Returns:
        th.Tensor: (n, n_tmpls) costs of using each template
    """
    # th.Tensor: (n, n_tmpls) costs of using each template
    n_covs: int = txs.shape[1]
    _device: th.device = inps.device
    inps = inps.to(device=device)
    txs = txs.to(device=device)
    tcels = tcels.to(device=device)
    costs_l: list[th.Tensor] = list()
    n_neighs = n_neighs + 1 if is_train else n_neighs
    for _inp in inps:
        # (n_covs, )
        _inp: th.Tensor
        _fm: th.Tensor = _inp[n_covs:]
        _fc: tuple[int, ...] = tuple(th.argwhere(_fm == 1).flatten().tolist())
        # (n_tmpls, n_covs)
        _fm_avail: th.Tensor = th.maximum(
            tmpls - _fm[None, :], th.as_tensor(0.0, device=device)
        )
        # (n_neighs, )
        _knnidxs: th.Tensor = mylib.ml.knn(
            xs=_inp[None, _fc].to(device=device),
            txs=txs[:, _fc].to(device=device),
            n_neighs=n_neighs,
            p=p,
        )[1][0].to(device="cpu")
        _knnidxs = _knnidxs[1:] if is_train else _knnidxs
        # (n_tmpls, )
        _cels: th.Tensor = th.mean(tcels[_knnidxs], dim=0)
        _costs: th.Tensor = _cels + lmbda * th.sum(_fm_avail, dim=1)
        costs_l.append(_costs)
    costs: th.Tensor = th.stack(costs_l, dim=0).to(device=_device)
    return costs


@th.no_grad()
def single_output_nnet_cost_est(
    inps: th.Tensor,
    nnet: th.nn.Module,
    lmbda: float,
    tmpls: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """single-output neural network cost estimator

    Args:
        inps (th.Tensor): (n, n_covs * 2)
        nnet (th.nn.Module): cross entropy neural net estimator
        inps (th.Tensor): (n, n_covs * 2)
        tmpls (th.Tensor): (n_tmpls, n_covs)
        device (th.device): device used to forward prop over nnet

    Returns:
        th.Tensor: (n, n_tmpls) costs of using each template
    """
    n_covs: int = tmpls.shape[1]
    nnet.eval().to(device=device)
    # (n_tmpls, n_covs)
    tmpls = tmpls.to(device=device)
    # (n, 2 * n_covs)
    inps = inps.to(device=device)
    # TODO might have to switch to looping over tmpl dim
    # (n, n_tmpls, n_covs)
    tmpls_: th.Tensor = tmpls[None, :, :].expand(len(inps), -1, -1)
    # (n, n_tmpls, 2 * n_covs)
    inps_: th.Tensor = inps[:, None, :].expand(-1, len(tmpls), -1)
    # (n * n_tmpls, 1)
    cels_: th.Tensor = nnet(th.cat((inps_, tmpls_), dim=2).flatten(0, 1))
    cels: th.Tensor = cels_[:, 0].unflatten(0, (len(inps), len(tmpls)))
    # (n, n_covs)
    fms: th.Tensor = inps[:, n_covs:].to(device=device)
    # (n, n_tmpls, n_covs)
    fms_avail: th.Tensor = th.maximum(
        tmpls[None, :, :] - fms[:, None, :], th.as_tensor(0.0, device=device)
    )
    # (n, n_tmpls)
    costs: th.Tensor = cels + lmbda * th.sum(fms_avail, dim=2)
    return costs


@th.no_grad()
def multi_output_nnet_cost_est(
    inps: th.Tensor,
    nnet: th.nn.Module,
    lmbda: float,
    tmpls: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """multi-output neural network cost estimator

    Args:
        inps (th.Tensor): (n, n_covs * 2)
        nnet (th.nn.Module): cross entropy neural net estimator
        inps (th.Tensor): (n, n_covs * 2)
        tmpls (th.Tensor): (n_tmpls, n_covs)
        device (th.device): device used to forward prop over nnet

    Returns:
        th.Tensor: (n, n_tmpls) costs of using each template
    """
    n_covs: int = tmpls.shape[1]
    nnet.eval().to(device=device)
    # (n_tmpls, n_covs)
    tmpls = tmpls.to(device=device)
    # (n, n_covs)
    inps = inps.to(device=device)
    fms: th.Tensor = inps[:, n_covs:]
    # (n, n_tmpls)
    cels: th.Tensor = nnet(inps)
    # (n, n_tmpls, n_covs)
    fms_avail: th.Tensor = th.maximum(
        tmpls[None, :, :] - fms[:, None, :], th.as_tensor(0.0, device=device)
    )
    costs: th.Tensor = cels + lmbda * th.sum(fms_avail, dim=2)
    return costs


@th.no_grad()
def selector_nnet_cost_est(
    inps: th.Tensor, nnet: th.nn.Module, device: th.device
) -> th.Tensor:
    """selector neural network cost estimator

    Args:
        inps (th.Tensor): (n, n_covs * 2)
        nnet (th.nn.Module): cross entropy neural net estimator
        device (th.device): device used to forward prop over nnet

    Returns:
        th.Tensor: (n, n_tmpls) costs of using each template
    """
    nnet.eval().to(device=device)
    # (n, n_covs)
    inps = inps.to(device=device)
    # (n, n_tmpls)
    costs: th.Tensor = -nnet(inps)
    return costs
