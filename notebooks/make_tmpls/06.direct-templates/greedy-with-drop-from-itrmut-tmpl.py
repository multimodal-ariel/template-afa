# %%
from __future__ import annotations

import copy
import os
from functools import partial
from typing import Optional

import _classifiers
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets
import mylib
import mymodels
import pandas as pd
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)


# %%
@th.no_grad()
def make_templates_direct_greedy_with_drop(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    tmpls_pt: Optional[th.Tensor],
    init_fidx: int,
    n_tmpls: int,
    max_features: Optional[int],
    lmbda: float,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
    n_neighs: int,
    vdata: Optional[thd.TensorDict],
    vclassifier: Optional[mymodels.classifiers.SubsetFeatureClassifier],
    metrics_func: thm.MetricCollection,
    eval_every_n_iter: int,
    ckpt_p: Optional[str] = None,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    generator = th.default_generator if generator is None else generator
    vclassifier = classifier if vclassifier is None else vclassifier
    if ckpt_p is not None:
        os.makedirs(ckpt_p, exist_ok=True)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # initialize templates
    tmpls: th.Tensor = (
        th.zeros((n_tmpls, n_covs), dtype=th.long) if tmpls_pt is None else tmpls_pt
    )
    tmpls[:, init_fidx] = 1
    if tmpls_pt is not None:
        assert n_tmpls == len(tmpls)
    # book keeping
    _nxt_blk_idx: int = 0 if tmpls_pt is None else n_tmpls - 1
    # direct greedy optimize
    pbar = tqdm.trange(n_iter, desc="direct-greedy", leave=False, dynamic_ncols=True)
    for _itr in pbar:
        # NOTE compute tpcomp for previous round of tmpls
        # (_ntmpls_prv, ncovs)
        _tmpls_prv: th.Tensor = tmpls[0 : _nxt_blk_idx + 1]
        # (ntdata, )
        _tpcomps_prv: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
            tmpls=_tmpls_prv,
            data=tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        # NOTE compute fitness of previous templates
        # (ntdata, )
        _rwds_prv: th.Tensor = th.max(_tpcomps_prv["rwds"], dim=1)[0]
        # () fitness_prv is a scaler
        _fitness_prv: th.Tensor = th.mean(_rwds_prv)
        # NOTE make all possible future templates without duplication
        _ntmpls: int = len(_tmpls_prv)
        _uctmpls_inds: set[frozenset[tuple[int, ...]]] = set()
        # for each template, generate "successor" collection of tmpls
        for _tidx in range(_ntmpls):
            # copy all templates except the one indexed by _tidx
            _base_inds: set[tuple[int, ...]] = {
                tuple(_t.tolist()) for _i, _t in enumerate(_tmpls_prv) if _i != _tidx
            }
            # given tidx-th template, for each possible location, toggle avaialability of one feature to tidx-th template
            _zeropos: list[int] = th.argwhere(_tmpls_prv[_tidx] == 0).flatten().tolist()
            for _pos in _zeropos:
                _cinds: set[tuple[int, ...]] = copy.deepcopy(_base_inds)
                _new_ind: th.Tensor = _tmpls_prv[_tidx].clone()
                _new_ind[_pos] = 1 if _new_ind == 0 else 0
                if th.sum(_new_ind) > max_features:
                    continue
                _cinds.add(tuple(_new_ind.tolist()))
                # filter out the case where the _new_ind is not something new to _base_inds
                if len(_cinds) == len(_base_inds):
                    continue
                # add the complete next potential template collection back to _tctmpls_inds
                _uctmpls_inds.add(frozenset(_cinds))
        # NOTE make list of potential future templates
        # a list of (_ncands, ) where each entry is (_ntmpls, )
        _ctmpls_l: list[th.Tensor] = [th.tensor([*_inds]) for _inds in _uctmpls_inds]
        # NOTE compute rewards
        # make unique feature combinations
        # keep track of all the information to recover the order
        _ctmpls_lens_l: list[int] = list(map(len, _ctmpls_l))
        _ctmpls_flt: th.Tensor
        _inv_idxs: th.Tensor
        _ctmpls_flt, _inv_idxs = th.unique(
            th.cat(_ctmpls_l, dim=0), dim=0, return_inverse=True
        )
        # list of (_ncands, ) each entry being a tensor of indices to _ctmpls_flt
        # the i-th entry of the list is the indices to which template is used to compose i-th candidate
        _inv_idxs_l: tuple[th.Tensor, ...] = th.split(_inv_idxs, _ctmpls_lens_l)
        # NOTE compute rewards
        # (ntdata, )
        _ctpcomps_flt: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
            tmpls=_ctmpls_flt,
            data=tdata,
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
            plf=plf,
        )
        # (ntdata, len(_ctmpls_flt))
        _crwds_flt: th.Tensor = _ctpcomps_flt["rwds"]
        # NOTE compute fitness
        # (_ncands, )
        _cfitness: th.Tensor = th.stack(
            [th.mean(th.max(_crwds_flt[:, _idxs], dim=1)[0]) for _idxs in _inv_idxs_l]
        )
        # NOTE compute improvements
        # (_ncands, )
        _improvements: th.Tensor = _cfitness - _fitness_prv
        _improvements = th.clamp(_improvements, min=0.0)
        # NOTE terminate loop if no imiprovement is made
        if not th.any(_improvements > 0.0) and _nxt_blk_idx + 1 >= n_tmpls:
            break
        # NOTE choose the one with the most improvement
        _slctd: int = int(th.argmax(_improvements, dim=0).item())
        tmpls[0 : _nxt_blk_idx + 1] = _ctmpls_l[_slctd]
        # NOTE if the blank template is modified, increment _nxt_blk_idx
        if th.all(th.sum(_ctmpls_l[_slctd], dim=1) != 1) and _nxt_blk_idx + 1 < n_tmpls:
            _nxt_blk_idx = _nxt_blk_idx + 1
        # NOTE log metrics
        _metrics_d: dict[str, float] = {
            "fitness": _cfitness[_slctd].item(),
            "improvement": _improvements[_slctd].item(),
        }
        pbar.set_postfix(_metrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(_metrics_d, "make-templates"), step=_itr
        )
        # NOTE keep track of validation set rollout performance
        if vdata is not None and _itr % eval_every_n_iter == 0:
            # (ntdata, _ncands)
            _ctcels: th.Tensor = _ctpcomps_flt["cels"][:, _inv_idxs_l[_slctd]]
            _vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib.functional.knn_cost_est(
                    x,
                    lmbda=lmbda,
                    txs=tdata["xs"],
                    tcels=_ctcels,
                    tmpls=_ctmpls_l[_slctd],
                    n_neighs=n_neighs,
                    p=2,
                ),
                init_fidx=init_fidx,
                tmpls=_ctmpls_l[_slctd],
                lmbda=lmbda,
                metrics_func=metrics_func,
                plf=plf,
            )
            plf.log_dict(mylib.utils.add_prefix_to_dict(_vmetrics_d, "val"), step=_itr)
        if ckpt_p is not None:
            th.save(tmpls, os.path.join(ckpt_p, f"tmpls_itr{_itr}.pt"))
    pbar.close()
    return tmpls


# %%
# NOTE big5
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
tmpl_run_p: Optional[str] = None
tmpls_pt: Optional[th.Tensor] = None
nnet_run_p: str = (
    "experiments/pretrain/nnet_subset_feature_classifier/outputs/big5/20250312_224514"
)
nnet_run_p = os.path.join(mylib.utils.get_project_root_dir(), nnet_run_p)
nnet_run_cfg = OmegaConf.load(os.path.join(nnet_run_p, ".hydra", "config.yaml"))
tclassifier = (
    mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
        nnet=hd.utils.instantiate(
            nnet_run_cfg.nnet,
            in_features=n_covs * 2,
            out_features=n_labels,
        ),
        xs_train=tdata["xs"].numpy(),
        ys_train=tdata["ys"].numpy(),
        fit_kwargs=hd.utils.instantiate(nnet_run_cfg.nnet_fit_cfg),
        state_dict_p=os.path.join(
            mylib.utils.get_project_root_dir(), nnet_run_p, "classifier.pt"
        ),
    )
)
tclassifier.fit_kwargs["n_iter"] = 1000
tclassifier.fit_kwargs["bsz"] = 8192
vclassifier = tclassifier
max_tdata: Optional[int] = None
init_fidx: int = 35
n_tmpls: int = 128
lmbda: float = 0.075
n_iter: int = 4096
n_neighs = 100
min_features: int = 1
max_features: Optional[int] = None
bsz: int = 409600

# %%
# NOTE this block is optional and is only needed if we want
tmpl_run_p = "experiments/make_template/outputs/big5_cnnet/20250314_112541/12"
tmpl_run_p = os.path.join(mylib.utils.get_project_root_dir(), tmpl_run_p)

# %%
# # NOTE charfont-1500
# data_name: str = "charfont-1500"
# _tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# n_covs: int = _tdata["xs"].shape[1]
# n_labels: int = len(th.unique(_tdata["ys"]))
# max_tdata: Optional[int] = 8192
# _tdata_shuffle_idxs = th.randperm(len(_tdata))
# tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
# extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
# run_p: str = (
#     "experiments/pretrain/nnet_subset_feature_classifier/outputs/charfont-1500/20250624_170501"
# )
# run_cfg = OmegaConf.load(
#     os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
# )
# tclassifier = (
#     mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
#         nnet=hd.utils.instantiate(
#             run_cfg.nnet,
#             in_features=n_covs * 2,
#             out_features=n_labels,
#         ),
#         xs_train=tdata["xs"].numpy(),
#         ys_train=tdata["ys"].numpy(),
#         fit_kwargs=hd.utils.instantiate(run_cfg.nnet_fit_cfg),
#         state_dict_p=os.path.join(
#             mylib.utils.get_project_root_dir(), run_p, "classifier.pt"
#         ),
#     )
# )
# tclassifier.fit_kwargs["n_iter"] = 10
# vclassifier = None
# init_fidx: int | None = 132
# n_tmpls_targ: int = 128
# n_cands_targ: int = 10_000
# lmbda: float = 0.075
# min_features_targ: int = 1
# max_features_targ: Optional[int] = None
# bsz: int = 8192

# %%
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "greedy-from-itrmut-tmpl")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")
ckpt_p: str = os.path.join(tfb_logger.log_dir, "checkpoints")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

# %%
if tmpl_run_p is not None:
    tmpls_pt = th.load(os.path.join(tmpl_run_p, "tmpls.pt"), weights_only=False)
    mktmpl_run_cfg = OmegaConf.load(os.path.join(tmpl_run_p, ".hydra", "config.yaml"))
    lmbda = mktmpl_run_cfg.lmbda
    n_neighs = mktmpl_run_cfg.n_neighs
    init_fidx = mktmpl_run_cfg.init_fidx

# %%
if init_fidx is None:
    init_fidx, bestfm = tafalib.makers.templates.identify_init_fidx(
        tdata=tdata,
        classifier=tclassifier,
        max_features=max_features,
        n_repeat=2,
        n_masks=500,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )

# %%
tmpls: th.Tensor = make_templates_direct_greedy_with_drop(
    tdata=tdata,
    classifier=tclassifier,
    tmpls_pt=tmpls_pt,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls,
    max_features=max_features,
    lmbda=lmbda,
    n_iter=n_iter,
    bsz=bsz,
    plf=plf,
    n_neighs=n_neighs,
    vdata=vdata,
    vclassifier=vclassifier,
    metrics_func=metrics_func,
    eval_every_n_iter=1,
    ckpt_p=ckpt_p,
)

# %%
# tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
#     tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz, plf=plf
# )

# # %%
# metrics_func.reset()
# snfobsd_l: list[int] = list()
# snfcomb_l: list[int] = list()
# for _data in vdata:
#     _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode(
#         x=_data["xs"],
#         classifier=vclassifier,
#         cost_est=lambda x: tafalib.functional.knn_cost_est(
#             x,
#             lmbda=lmbda,
#             txs=tdata["xs"],
#             tcels=tpcomp["cels"],
#             tmpls=tmpls,
#             n_neighs=10,
#             p=2,
#         ),
#         init_fidx=init_fidx,
#         tmpls=tmpls,
#         plf=plf,
#     )
#     snfobsd_l.append(len(_fobsd_l))
#     snfcomb_l.append(len(_fcomb))
#     metrics_func.update(
#         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
#     )
# metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# metrics_func.reset()
# metrics_d.update(
#     {
#         "init_fidx": init_fidx,
#         "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
#         "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
#     }
# )
# print(pd.Series(metrics_d))


# # %%
# metrics_func.reset()
# snfobsd_l: list[int] = list()
# snfcomb_l: list[int] = list()
# for _data in vdata:
#     _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
#         x=_data["xs"],
#         classifier=vclassifier,
#         cost_est=lambda x: tafalib.functional.knn_cost_est(
#             x,
#             lmbda=lmbda,
#             txs=tdata["xs"],
#             tcels=tpcomp["cels"],
#             tmpls=tmpls,
#             n_neighs=10,
#             p=2,
#         ),
#         init_fidx=init_fidx,
#         tmpls=tmpls,
#         plf=plf,
#     )
#     snfobsd_l.append(len(_fobsd_l))
#     snfcomb_l.append(len(_fcomb))
#     metrics_func.update(
#         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
#     )
# metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# metrics_func.reset()
# metrics_d.update(
#     {
#         "init_fidx": init_fidx,
#         "feature used & observed": th.mean(
#             th.as_tensor(snfobsd_l, dtype=th.float32)
#         ).item(),
#     }
# )
# print(pd.Series(metrics_d))

# # %%
# metrics_func.reset()
# snfobsd_l: list[int] = list()
# snfcomb_l: list[int] = list()
# for _data in vdata:
#     _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
#         x=_data["xs"],
#         classifier=tclassifier,
#         cost_est=lambda x: tafalib.functional.knn_cost_est(
#             x,
#             lmbda=lmbda,
#             txs=tdata["xs"],
#             tcels=tpcomp["cels"],
#             tmpls=tmpls,
#             n_neighs=10,
#             p=2,
#         ),
#         init_fidx=init_fidx,
#         tmpls=tmpls,
#         plf=plf,
#     )
#     snfobsd_l.append(len(_fobsd_l))
#     snfcomb_l.append(len(_fcomb))
#     metrics_func.update(
#         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
#     )
# metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# metrics_func.reset()
# metrics_d.update(
#     {
#         "init_fidx": init_fidx,
#         "feature used & observed": th.mean(
#             th.as_tensor(snfobsd_l, dtype=th.float32)
#         ).item(),
#     }
# )
# print(pd.Series(metrics_d))

# %%
