from __future__ import annotations

import mylib.ml
import torch as th


def knn_cost_est(
    inps: th.Tensor,
    lmbda: float,
    txs: th.Tensor,
    tcels: th.Tensor,
    tmpls: th.Tensor,
    n_neighs: int,
    p: float = 2,
    is_train: bool = False,
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
    device: th.device = inps.device
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
        _fm_avail: th.Tensor = th.maximum(tmpls - _fm[None, :], th.as_tensor(0.0))
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
    costs: th.Tensor = th.stack(costs_l, dim=0).to(device=device)
    return costs
