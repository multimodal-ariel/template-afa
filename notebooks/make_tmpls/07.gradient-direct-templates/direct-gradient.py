# %%
from __future__ import annotations

import copy
import os
from functools import partial
from typing import Callable, Iterable, Optional

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
import torch.distributions.utils
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)


# %%
@th.no_grad()
def make_templates_direct_gradient(
    tdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    tmpls_pt: Optional[th.Tensor],
    init_fidx: int,
    n_tmpls: int,
    max_features: Optional[int],
    lmbda: float,
    make_opt_fn: Callable[[Iterable[th.Tensor]], th.optim.Optimizer],
    n_iter: int,
    n_iter_gradient_accumulate: int,
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
    # initialize
    assert isinstance(
        classifier, mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier
    )
    generator = th.default_generator if generator is None else generator
    vclassifier = classifier if vclassifier is None else vclassifier
    assert isinstance(
        vclassifier, mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier
    )
    if ckpt_p is not None:
        os.makedirs(ckpt_p, exist_ok=True)
    classifier = classifier.to(device=plf.device)
    vclassifier = vclassifier.to(device=plf.device)
    # record shapes
    n_data: int = len(tdata)
    n_covs: int = tdata["xs"].shape[1]
    max_features = n_covs if max_features is None else max_features
    # initialize templates in logit space
    ltmpls: th.Tensor = (
        th.nn.init.orthogonal_(
            th.empty(
                (n_tmpls, n_covs),
                dtype=th.float32,
                device=plf.device,
            )
        )
        if tmpls_pt is None
        else torch.distributions.utils.probs_to_logits(
            th.clamp(tmpls_pt, 0.3, 0.7), is_binary=True
        ).to(device=plf.device)
    )
    ltmpls[:, init_fidx] = th.max(ltmpls)
    ltmpls = ltmpls.requires_grad_(True)
    # make optimizer
    opt: th.optim.Optimizer = make_opt_fn([ltmpls])
    # direct greedy optimize
    if tmpls_pt is not None:
        assert n_tmpls == len(ltmpls)
    pbar = tqdm.trange(n_iter, desc="direct-gradient", leave=False, dynamic_ncols=True)
    for _itr in pbar:
        # NOTE draw a batch of training instances
        _bidxs: th.Tensor = (
            th.multinomial(
                th.ones(n_data),
                num_samples=bsz,
                replacement=False,
                generator=generator,
            )
            if bsz < n_data
            else th.arange(n_data)
        )
        _bsz: int = len(_bidxs)
        _btdata = tdata[_bidxs]
        # (_btdata, n_covs)
        _btxs: th.Tensor = _btdata["xs"]
        # (_btdata, )
        _btys: th.Tensor = _btdata["ys"]
        # NOTE draw a random sample from loogit template
        # TODO anneal temperature as it iterates
        _btmpls: th.Tensor = th.distributions.RelaxedBernoulli(
            temperature=th.tensor(1.0, device=plf.device), logits=ltmpls
        ).rsample()

        # NOTE compute cross entropy loss for each instance and template combination
        _brwds_l: list[th.Tensor]

        def _minibatch_compute_oracle_rwd(_bbidxs: th.Tensor) -> th.Tensor:
            # (_bsz, n_covs)
            _bbctxs: th.Tensor = _btxs[_bbidxs[:, 0], :].to(device=plf.device)
            _bbacts: th.Tensor = _btmpls[_bbidxs[:, 1], :].to(device=plf.device)
            # (_bsz, n_labels)
            _bbpyhats: th.Tensor = classifier.predict_proba(_bbctxs, _bbacts)
            # (_bsz, )
            _bbcels: th.Tensor = th.nn.functional.cross_entropy(
                _bbpyhats,
                _btys[_bbidxs[:, 0]].to(device=plf.device),
                reduction="none",
            )
            _bbrwds: th.Tensor = -_bbcels - lmbda * th.sum(_bbacts, dim=1)
            return _bbrwds

        with th.autograd.graph.save_on_cpu():
            _brwds_l = [
                _minibatch_compute_oracle_rwd(_bbidxs)
                for _bbidxs in th.split(
                    th.cartesian_prod(th.arange(_bsz), th.arange(n_tmpls)), bsz
                )
            ]
        # (_bsz, n_tmpls)
        _brwds: th.Tensor = th.unflatten(
            th.cat(_brwds_l, dim=0), dim=0, sizes=(_bsz, n_tmpls)
        )
        # NOTE compute costs
        # (_bsz, n_tmpls)
        _bweights: th.Tensor = th.distributions.RelaxedOneHotCategorical(
            temperature=th.tensor(1.0, device=plf.device), logits=_brwds
        ).rsample()
        # (_bsz, )
        _bcosts: th.Tensor = th.mean(-_brwds * _bweights, dim=1)
        # compute losss
        _bloss: th.Tensor = th.mean(_bcosts)
        # back prop and update ltmpls when needed
        _bloss.backward()
        if _itr % n_iter_gradient_accumulate == 0:
            opt.step()
            opt.zero_grad()
        # NOTE log metrics
        _metrics_d: dict[str, float] = {
            "loss": _bloss.item(),
        }
        pbar.set_postfix(_metrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(_metrics_d, "direct-gradient"), step=_itr
        )
        # NOTE keep track of validation set rollout performance
        if vdata is not None and _itr % eval_every_n_iter == 0:
            _tmpls: th.Tensor = th.where(th.sigmoid(ltmpls) < 0.5, 0.0, 1.0)
            _tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
                _tmpls, data=tdata, classifier=classifier, lmbda=lmbda, bsz=bsz, plf=plf
            )
            _vmetrics_d: dict[str, float] = tafalib.utils.evaluate(
                data=vdata,
                classifier=vclassifier,
                cost_est=lambda x: tafalib.functional.knn_cost_est(
                    x,
                    lmbda=lmbda,
                    txs=tdata["xs"],
                    tcels=_tpcomp["cels"],
                    tmpls=_tmpls,
                    n_neighs=n_neighs,
                    p=2,
                ),
                init_fidx=init_fidx,
                tmpls=_tmpls,
                lmbda=lmbda,
                metrics_func=metrics_func,
                plf=plf,
            )
            plf.log_dict(mylib.utils.add_prefix_to_dict(_vmetrics_d, "val"), step=_itr)
        if ckpt_p is not None:
            th.save(ltmpls, os.path.join(ckpt_p, f"ltmpls_itr{_itr}.pt"))
    pbar.close()
    return ltmpls


# %%
# # NOTE big5
# data_name: str = "big5_C_cls"
# _tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# n_covs: int = _tdata["xs"].shape[1]
# n_labels: int = len(th.unique(_tdata["ys"]))
# _tdata_shuffle_idxs = th.randperm(len(_tdata))
# tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
# extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
# tmpl_run_p: Optional[str] = None
# tmpls_pt: Optional[th.Tensor] = None
# nnet_run_p: str = (
#     "experiments/pretrain/nnet_subset_feature_classifier/outputs/big5/20250312_224514"
# )
# nnet_run_p = os.path.join(mylib.utils.get_project_root_dir(), nnet_run_p)
# nnet_run_cfg = OmegaConf.load(os.path.join(nnet_run_p, ".hydra", "config.yaml"))
# tclassifier = (
#     mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
#         nnet=hd.utils.instantiate(
#             nnet_run_cfg.nnet,
#             in_features=n_covs * 2,
#             out_features=n_labels,
#         ),
#         xs_train=tdata["xs"].numpy(),
#         ys_train=tdata["ys"].numpy(),
#         fit_kwargs=hd.utils.instantiate(nnet_run_cfg.nnet_fit_cfg),
#         state_dict_p=os.path.join(
#             mylib.utils.get_project_root_dir(), nnet_run_p, "classifier.pt"
#         ),
#     )
# )
# tclassifier.fit_kwargs["n_iter"] = 1000
# tclassifier.fit_kwargs["bsz"] = 8192
# vclassifier = tclassifier
# max_tdata: Optional[int] = None
# init_fidx: int = 35
# n_tmpls: int = 128
# lmbda: float = 0.075
# n_iter: int = 4096
# n_neighs = 100
# min_features: int = 1
# max_features: Optional[int] = None
# bsz: int = 409600

# # %%
# # NOTE this block is optional and is only needed if we want
# tmpl_run_p = "experiments/make_template/outputs/big5_cnnet/20250314_112541/12"
# tmpl_run_p = os.path.join(mylib.utils.get_project_root_dir(), tmpl_run_p)

# # %%
# # # NOTE charfont-1500
# # data_name: str = "charfont-1500"
# # _tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# # n_covs: int = _tdata["xs"].shape[1]
# # n_labels: int = len(th.unique(_tdata["ys"]))
# # max_tdata: Optional[int] = 8192
# # _tdata_shuffle_idxs = th.randperm(len(_tdata))
# # tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
# # extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
# # run_p: str = (
# #     "experiments/pretrain/nnet_subset_feature_classifier/outputs/charfont-1500/20250624_170501"
# # )
# # run_cfg = OmegaConf.load(
# #     os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
# # )
# # tclassifier = (
# #     mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
# #         nnet=hd.utils.instantiate(
# #             run_cfg.nnet,
# #             in_features=n_covs * 2,
# #             out_features=n_labels,
# #         ),
# #         xs_train=tdata["xs"].numpy(),
# #         ys_train=tdata["ys"].numpy(),
# #         fit_kwargs=hd.utils.instantiate(run_cfg.nnet_fit_cfg),
# #         state_dict_p=os.path.join(
# #             mylib.utils.get_project_root_dir(), run_p, "classifier.pt"
# #         ),
# #     )
# # )
# # tclassifier.fit_kwargs["n_iter"] = 10
# # vclassifier = None
# # init_fidx: int | None = 132
# # n_tmpls_targ: int = 128
# # n_cands_targ: int = 10_000
# # lmbda: float = 0.075
# # min_features_targ: int = 1
# # max_features_targ: Optional[int] = None
# # bsz: int = 8192

# # %%
# metrics_func = thm.MetricCollection(
#     {
#         "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
#         "precision": thm.Precision(task="multiclass", num_classes=n_labels),
#         "recall": thm.Recall(task="multiclass", num_classes=n_labels),
#         "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
#         "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
#     }
# )

# # %%
# # configure logger and ckpt path
# output_dir: str = os.path.join("outputs", "run", data_name, "greedy-from-itrmut-tmpl")
# os.makedirs(output_dir, exist_ok=True)
# tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
# csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")
# ckpt_p: str = os.path.join(tfb_logger.log_dir, "checkpoints")

# # %%
# plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

# # %%
# if tmpl_run_p is not None:
#     tmpls_pt = th.load(os.path.join(tmpl_run_p, "tmpls.pt"), weights_only=False)
#     mktmpl_run_cfg = OmegaConf.load(os.path.join(tmpl_run_p, ".hydra", "config.yaml"))
#     lmbda = mktmpl_run_cfg.lmbda
#     n_neighs = mktmpl_run_cfg.n_neighs
#     init_fidx = mktmpl_run_cfg.init_fidx

# # %%
# if init_fidx is None:
#     init_fidx, bestfm = tafalib.makers.templates.identify_init_fidx(
#         tdata=tdata,
#         classifier=tclassifier,
#         max_features=max_features,
#         n_repeat=2,
#         n_masks=500,
#         lmbda=lmbda,
#         bsz=bsz,
#         plf=plf,
#     )

# # %%
# tmpls: th.Tensor = make_templates_direct_greedy(
#     tdata=tdata,
#     classifier=tclassifier,
#     tmpls_pt=tmpls_pt,
#     init_fidx=init_fidx,
#     n_tmpls=n_tmpls,
#     max_features=max_features,
#     lmbda=lmbda,
#     n_iter=n_iter,
#     bsz=bsz,
#     plf=plf,
#     n_neighs=n_neighs,
#     vdata=vdata,
#     vclassifier=vclassifier,
#     metrics_func=metrics_func,
#     eval_every_n_iter=1,
#     ckpt_p=ckpt_p,
# )

# # %%
# # tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
# #     tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz, plf=plf
# # )

# # # %%
# # metrics_func.reset()
# # snfobsd_l: list[int] = list()
# # snfcomb_l: list[int] = list()
# # for _data in vdata:
# #     _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode(
# #         x=_data["xs"],
# #         classifier=vclassifier,
# #         cost_est=lambda x: tafalib.functional.knn_cost_est(
# #             x,
# #             lmbda=lmbda,
# #             txs=tdata["xs"],
# #             tcels=tpcomp["cels"],
# #             tmpls=tmpls,
# #             n_neighs=10,
# #             p=2,
# #         ),
# #         init_fidx=init_fidx,
# #         tmpls=tmpls,
# #         plf=plf,
# #     )
# #     snfobsd_l.append(len(_fobsd_l))
# #     snfcomb_l.append(len(_fcomb))
# #     metrics_func.update(
# #         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
# #     )
# # metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# # metrics_func.reset()
# # metrics_d.update(
# #     {
# #         "init_fidx": init_fidx,
# #         "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
# #         "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
# #     }
# # )
# # print(pd.Series(metrics_d))


# # # %%
# # metrics_func.reset()
# # snfobsd_l: list[int] = list()
# # snfcomb_l: list[int] = list()
# # for _data in vdata:
# #     _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
# #         x=_data["xs"],
# #         classifier=vclassifier,
# #         cost_est=lambda x: tafalib.functional.knn_cost_est(
# #             x,
# #             lmbda=lmbda,
# #             txs=tdata["xs"],
# #             tcels=tpcomp["cels"],
# #             tmpls=tmpls,
# #             n_neighs=10,
# #             p=2,
# #         ),
# #         init_fidx=init_fidx,
# #         tmpls=tmpls,
# #         plf=plf,
# #     )
# #     snfobsd_l.append(len(_fobsd_l))
# #     snfcomb_l.append(len(_fcomb))
# #     metrics_func.update(
# #         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
# #     )
# # metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# # metrics_func.reset()
# # metrics_d.update(
# #     {
# #         "init_fidx": init_fidx,
# #         "feature used & observed": th.mean(
# #             th.as_tensor(snfobsd_l, dtype=th.float32)
# #         ).item(),
# #     }
# # )
# # print(pd.Series(metrics_d))

# # # %%
# # metrics_func.reset()
# # snfobsd_l: list[int] = list()
# # snfcomb_l: list[int] = list()
# # for _data in vdata:
# #     _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
# #         x=_data["xs"],
# #         classifier=tclassifier,
# #         cost_est=lambda x: tafalib.functional.knn_cost_est(
# #             x,
# #             lmbda=lmbda,
# #             txs=tdata["xs"],
# #             tcels=tpcomp["cels"],
# #             tmpls=tmpls,
# #             n_neighs=10,
# #             p=2,
# #         ),
# #         init_fidx=init_fidx,
# #         tmpls=tmpls,
# #         plf=plf,
# #     )
# #     snfobsd_l.append(len(_fobsd_l))
# #     snfcomb_l.append(len(_fcomb))
# #     metrics_func.update(
# #         _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
# #     )
# # metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
# # metrics_func.reset()
# # metrics_d.update(
# #     {
# #         "init_fidx": init_fidx,
# #         "feature used & observed": th.mean(
# #             th.as_tensor(snfobsd_l, dtype=th.float32)
# #         ).item(),
# #     }
# # )
# # print(pd.Series(metrics_d))

# %%
