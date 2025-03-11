from __future__ import annotations

import torch as th


@th.no_grad()
def knn(
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
