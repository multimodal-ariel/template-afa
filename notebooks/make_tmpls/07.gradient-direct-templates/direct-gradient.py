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
def simple_exponential_decay_temperature(
    itr: int,
    decay_every_n_iter=100,
    init_temperature=1.0,
    rate=0.5,
    min_temperature: float = 1e-7,
) -> float:
    num_steps = itr // decay_every_n_iter
    temperature: float = init_temperature * (rate**num_steps)
    if temperature < min_temperature:
        temperature = min_temperature
    return temperature


# %%
def collision_probability_penalty(ltmpls: th.Tensor) -> th.Tensor:
    """
    Penalize the probability that two rows will be identical in discrete space.

    Args:
        logits: (M, D) tensor of logits
        lambda_reg: penalty strength

    Mathematical intuition:
    - For binary variables, P(bit_i == bit_j) = p_i * p_j + (1-p_i) * (1-p_j)
    - For entire rows to be identical, ALL bits must match
    - P(row_i == row_j) = ∏_d P(bit_i[d] == bit_j[d])
    """
    P = th.sigmoid(ltmpls)  # Convert logits to probabilities [0,1]
    M, D = P.shape
    # Expand dimensions for broadcasting
    # P_i: (M, 1, D) - each row i broadcast across dimension 1
    # P_j: (1, M, D) - each row j broadcast across dimension 0
    P_i = P.unsqueeze(1)  # Shape: (M, 1, D)
    P_j = P.unsqueeze(0)  # Shape: (1, M, D)
    # For each pair (i,j) and each dimension d:
    # P(bit_i[d] == bit_j[d]) = P_i[d] * P_j[d] + (1-P_i[d]) * (1-P_j[d])
    same_bit_prob = P_i * P_j + (1 - P_i) * (1 - P_j)
    # Shape: (M, M, D)
    # P(row_i == row_j) = ∏_d P(bit_i[d] == bit_j[d])
    # In log space: log(∏) = ∑log, then exp back
    # But torch.prod works directly on small probabilities
    collision_prob = th.prod(same_bit_prob, dim=-1)  # Shape: (M, M)
    # Remove diagonal (self-comparisons) and avoid double-counting
    # Upper triangular mask excludes diagonal and lower triangle
    mask = th.triu(th.ones(M, M, device=ltmpls.device), diagonal=1)
    return th.sum(collision_prob * mask)


def expected_hamming_penalty(ltmpls: th.Tensor) -> th.Tensor:
    """
    Maximize expected Hamming distance between all pairs of rows.

    Mathematical intuition:
    - Hamming distance = number of differing bits
    - E[HD(row_i, row_j)] = ∑_d P(bit_i[d] ≠ bit_j[d])
    - P(bit_i[d] ≠ bit_j[d]) = p_i*(1-p_j) + (1-p_i)*p_j
    """
    P = th.sigmoid(ltmpls)
    M, D = P.shape
    P_i = P.unsqueeze(1)  # (M, 1, D)
    P_j = P.unsqueeze(0)  # (1, M, D)
    # P(bit differs) = P_i * (1-P_j) + (1-P_i) * P_j
    # This is the probability that bit i is 1 and bit j is 0, OR
    # bit i is 0 and bit j is 1
    diff_bit_prob = P_i * (1 - P_j) + (1 - P_i) * P_j
    # Shape: (M, M, D)
    # Expected Hamming distance = sum over all dimensions
    expected_hamming = th.sum(diff_bit_prob, dim=-1)  # Shape: (M, M)
    # We want to MAXIMIZE Hamming distance, so negative penalty
    mask = th.triu(th.ones(M, M, device=ltmpls.device), diagonal=1)
    # Negative because we want to maximize distance (minimize negative distance)
    return -th.sum(expected_hamming * mask)


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
    bsz_compute_minibatch_cost: int,
    alpha_unq: float,
    get_temperature_fn: Callable[[int], float],
    compute_uniqueness_penalty_fn: Callable[[th.Tensor], th.Tensor],
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
    opt.zero_grad()
    # direct greedy optimize
    if tmpls_pt is not None:
        assert n_tmpls == len(ltmpls)
    pbar = tqdm.trange(n_iter, desc="direct-gradient", leave=False, dynamic_ncols=True)
    for _itr in pbar:
        _temperature: float = get_temperature_fn(_itr)
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
            temperature=th.tensor(_temperature, device=plf.device), logits=ltmpls
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
                    th.cartesian_prod(th.arange(_bsz), th.arange(n_tmpls)),
                    bsz_compute_minibatch_cost,
                )
            ]
        # (_bsz, n_tmpls)
        _brwds: th.Tensor = th.unflatten(
            th.cat(_brwds_l, dim=0), dim=0, sizes=(_bsz, n_tmpls)
        )
        # NOTE compute costs
        # (_bsz, n_tmpls)
        _bweights: th.Tensor = th.distributions.RelaxedOneHotCategorical(
            temperature=th.tensor(_temperature, device=plf.device), logits=_brwds
        ).rsample()
        # (_bsz, )
        _bcosts: th.Tensor = th.mean(-_brwds * _bweights, dim=1)
        _bunq_loss: th.Tensor = compute_uniqueness_penalty_fn(ltmpls)
        # compute penalties
        # TODO encourage sparsity, encourage pairwise independence
        # compute loss
        _bloss: th.Tensor = th.mean(_bcosts) + alpha_unq * _bunq_loss
        # back prop and update ltmpls when needed
        _bloss.backward()
        if (_itr + 1) % n_iter_gradient_accumulate == 0 or (_itr + 1) == n_iter:
            opt.step()
            opt.zero_grad()
        # NOTE log metrics
        _metrics_d: dict[str, float] = {
            "temperature": _temperature,
            "cost": th.mean(_bcosts).item(),
            "uniqueness_loss": _bunq_loss.item(),
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
    tmpls: th.Tensor = th.sigmoid(ltmpls).to(dtype=th.int64)
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
lr: float = 1e-3
lmbda: float = 0.075
n_iter: int = 10_000
n_iter_gradient_accumulate: int = 500
n_neighs = 100
alpha_unq: float = 0.1
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
tmpls: th.Tensor = make_templates_direct_gradient(
    tdata=tdata,
    classifier=tclassifier,
    tmpls_pt=tmpls_pt,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls,
    max_features=max_features,
    lmbda=lmbda,
    make_opt_fn=lambda _p: th.optim.Adam(_p, lr=lr),
    n_iter=n_iter,
    n_iter_gradient_accumulate=n_iter_gradient_accumulate,
    bsz=bsz,
    bsz_compute_minibatch_cost=bsz,
    get_temperature_fn=simple_exponential_decay_temperature,
    alpha_unq=alpha_unq,
    compute_uniqueness_penalty_fn=collision_probability_penalty,
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
