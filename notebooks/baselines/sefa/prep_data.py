# %%
from __future__ import annotations

import os
from typing import Any

import mydatasets
import sefalib
import tensordict as thd
import torch as th


# %%
def _preprocess_data(
    data_dict: dict[str, Any],
    tdata: thd.TensorDict,
    vdata: thd.TensorDict,
    tstdata: thd.TensorDict,
    num_bins: int = 100,
    size_normal: float = 0.0,
    ratio_uniform: float = 0.0,
    rseed: int = 279,
) -> tuple[thd.TensorDict, thd.TensorDict, thd.TensorDict]:
    n_covs: int = tdata["xs"].shape[1]
    n_cont_feats: int = data_dict["num_con_features"]
    tms: th.Tensor = tdata["ms"] if "ms" in tdata else th.ones_like(tdata["xs"])
    vms: th.Tensor = vdata["ms"] if "ms" in vdata else th.ones_like(vdata["xs"])
    tstms: th.Tensor = tstdata["ms"] if "ms" in tstdata else th.ones_like(tstdata["xs"])
    # process continuous covariates
    # transform continuous covariates
    empirical_cdf = sefalib.data.preprocessing_utils.VectorEmpiricalCDF(
        num_bins=num_bins,
        size_normal=size_normal,
        ratio_uniform=ratio_uniform,
        rseed=rseed,
    )
    empirical_cdf.fit(tdata["xs"][:, :n_cont_feats], tms[:, :n_cont_feats])
    txs_cdf: th.Tensor = empirical_cdf.transform(
        tdata["xs"][:, :n_cont_feats], tms[:, :n_cont_feats]
    )
    vxs_cdf: th.Tensor = empirical_cdf.transform(
        vdata["xs"][:, :n_cont_feats], vms[:, :n_cont_feats]
    )
    tstxs_cdf: th.Tensor = empirical_cdf.transform(
        tstdata["xs"][:, :n_cont_feats], tstms[:, :n_cont_feats]
    )
    # standardize continuous features
    standardizer = sefalib.data.preprocessing_utils.Standardizer()
    standardizer.fit(tdata["xs"][:, :n_cont_feats], tms[:, :n_cont_feats])
    txs_std: th.Tensor = standardizer.transform(
        tdata["xs"][:, :n_cont_feats], tms[:, :n_cont_feats]
    )
    vxs_std: th.Tensor = standardizer.transform(
        vdata["xs"][:, :n_cont_feats], vms[:, :n_cont_feats]
    )
    tstxs_std: th.Tensor = standardizer.transform(
        tstdata["xs"][:, :n_cont_feats], tstms[:, :n_cont_feats]
    )
    # if contains discrete features, concat them back
    if n_covs > n_cont_feats:
        txs_cdf = th.cat((txs_cdf, tdata["xs"][:, n_cont_feats:]), dim=1)
        txs_std = th.cat((txs_std, tdata["xs"][:, n_cont_feats:]), dim=1)
        vxs_cdf = th.cat((vxs_cdf, vdata["xs"][:, n_cont_feats:]), dim=1)
        vxs_std = th.cat((vxs_std, vdata["xs"][:, n_cont_feats:]), dim=1)
        tstxs_cdf = th.cat((tstxs_cdf, tstdata["xs"][:, n_cont_feats:]), dim=1)
        tstxs_std = th.cat((tstxs_std, tstdata["xs"][:, n_cont_feats:]), dim=1)
    tdata_ = tdata.update(
        {
            "xs_cdf": txs_cdf,
            "xs_std": txs_std,
            "ms": tms,
        },
        inplace=False,
    )
    vdata_ = vdata.update(
        {
            "xs_cdf": vxs_cdf,
            "xs_std": vxs_std,
            "ms": vms,
        },
        inplace=False,
    )
    tstdata_ = tstdata.update(
        {
            "xs_cdf": tstxs_cdf,
            "xs_std": tstxs_std,
            "ms": tstms,
        },
        inplace=False,
    )
    return tdata_, vdata_, tstdata_


# %%
sefa_p: str = os.path.join(mydatasets.common.get_datasets_files_root_dir(), "sefa")
os.makedirs(sefa_p, exist_ok=True)

# %%
tcube, vcube, tstcube = mydatasets.aaco.load_aaco_data(
    "cube_20_0.3", to_normalize=False
)
print(tcube["xs"].shape)
cube_dict = {
    "num_con_features": tcube["xs"].shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": len(th.unique(tcube["ys"])),
    "metric": "accuracy",
    "max_dim": None,
}
tcube_, vcube_, tstcube_ = _preprocess_data(
    data_dict=cube_dict,
    tdata=tcube,
    vdata=vcube,
    tstdata=tstcube,
    num_bins=200,
    size_normal=0.0,
    ratio_uniform=0.0,
)
th.save(
    {
        "train": tcube_,
        "valid": vcube_,
        "test": tstcube_,
        "dataset_dict": cube_dict,
    },
    os.path.join(sefa_p, "cube_20_0.3.pt"),
)

# %%
tgas, vgas, tstgas = mydatasets.aaco.load_aaco_data("gas")
print(tgas["xs"].shape)
gas_dict = {
    "num_con_features": tgas["xs"].shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": len(th.unique(tgas["ys"])),
    "metric": "accuracy",
    "max_dim": None,
}
tgas_, vgas_, tstgas_ = _preprocess_data(
    data_dict=gas_dict,
    tdata=tgas,
    vdata=vgas,
    tstdata=tstgas,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
)
th.save(
    {
        "train": tgas_,
        "valid": vgas_,
        "test": tstgas_,
        "dataset_dict": gas_dict,
    },
    os.path.join(sefa_p, "gas.pt"),
)


# %%
tgrid, vgrid, tstgrid = mydatasets.aaco.load_aaco_data("grid_data")
print(tgrid["xs"].shape)
grid_dict = {
    "num_con_features": tgrid["xs"].shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": len(th.unique(tgrid["ys"])),
    "metric": "accuracy",
    "max_dim": None,
}
tgrid_, vgrid_, tstgrid_ = _preprocess_data(
    data_dict=grid_dict,
    tdata=tgrid,
    vdata=vgrid,
    tstdata=tstgrid,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
)
th.save(
    {
        "train": tgrid_,
        "valid": vgrid_,
        "test": tstgrid_,
        "dataset_dict": grid_dict,
    },
    os.path.join(sefa_p, "grid_data.pt"),
)

# %%
tmnist, vmnist, tstmnist = mydatasets.aaco.load_aaco_data("mnist")
print(tmnist["xs"].shape, tmnist["ys"].shape)
mnist_dict = {
    "num_con_features": tmnist["xs"].shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": len(th.unique(tmnist["ys"])),
    "metric": "accuracy",
    "max_dim": None,
}
tmnist_, vmnist_, tstmnist_ = _preprocess_data(
    data_dict=mnist_dict,
    tdata=tmnist,
    vdata=vmnist,
    tstdata=tstmnist,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.2,
)
th.save(
    {
        "train": tmnist_,
        "valid": vmnist_,
        "test": tstmnist_,
        "dataset_dict": mnist_dict,
    },
    os.path.join(sefa_p, "mnist.pt"),
)

# %%
tbg5, vbg5, tstbg5 = mydatasets.aaco.load_aaco_data("big5_C_cls", to_normalize=True)
print(tbg5["xs"].shape)
bg5_dict = {
    "num_con_features": tbg5["xs"].shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": len(th.unique(tbg5["ys"])),
    "metric": "accuracy",
    "max_dim": None,
}
tbg5_, vbg5_, tstbg5_ = _preprocess_data(
    data_dict=bg5_dict,
    tdata=tbg5,
    vdata=vbg5,
    tstdata=tstbg5,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
)
th.save(
    {
        "train": tbg5_,
        "valid": vbg5_,
        "test": tstbg5_,
        "dataset_dict": bg5_dict,
    },
    os.path.join(sefa_p, "big5_C_cls.pt"),
)


# %%
# tcharfont, vcharfont, tstcharfont = mydatasets.aaco.load_aaco_data(
#     "charfont-1500", to_normalize=True
# )
# print(tcharfont["xs"].shape)
# charfont_dict = {
#     "num_con_features": tcharfont["xs"].shape[1],
#     "num_cat_features": 0,
#     "most_categories": 0,
#     "out_dim": len(th.unique(tcharfont["ys"])),
#     "metric": "accuracy",
#     "max_dim": None,
# }
# tcharfont_, vcharfont_, tstcharfont_ = _preprocess_data(
#     data_dict=charfont_dict,
#     tdata=tcharfont,
#     vdata=vcharfont,
#     tstdata=tstcharfont,
#     num_bins=200,
#     size_normal=1e-5,
#     ratio_uniform=0.05,
# )
# th.save(
#     {
#         "train": tcharfont_,
#         "valid": vcharfont_,
#         "test": tstcharfont_,
#         "dataset_dict": charfont_dict,
#     },
#     os.path.join(sefa_p, "charfont-1500.pt"),
# )

# # %%
# tvolvo, vvolvo, tstvolvo = mydatasets.aaco.load_aaco_data("volvo", to_normalize=True)
# print(tvolvo["xs"].shape)

# %%
