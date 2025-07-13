# %%
from __future__ import annotations

import os
from typing import Optional

import _classifiers
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets
import mymodels
import networkx as nx
import pandas as pd
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
PROJ_ROOT: str = "../../../"


# %%
# NOTE make templates from scratch
@th.no_grad()
def make_corr_templates(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    to_update_classifier: bool,
    init_fidx: int,
    n_tmpls: int,
    n_cands: int,
    min_features: int,
    max_features: Optional[int],
    corr_thrsh: float,
    lmbda: float,
    bsz: int,
    plf: pl.Fabric,
) -> th.Tensor:
    """make templates greedily

    Args:
        tdata (thd.TensorDict): (n, ) training data
        max_tdata (Optional[int]): optional maximum training data to subsample if training data is too large and takes too long to complete
        classifier (mymodels.classifiers.SubsetFeatureClassifier): a subset feature classifier to use
        to_update_classifier (bool): to update subset feature classifier when a new collection of templates is generated and before the final collection of templates is found.
        init_fidx (int): initial feature index
        n_tmpls (int): number of templates to return
        n_cands (int): number of candidates used to search
        min_features (int): minimum number of features to include in each template; must be greater than 1.
        max_features (Optional[int]): maximum number of features to include in each template; if `None`, `n_covs` is used.
        lmbda (float): penalty term for choosing more features.
        bsz (int): batch size for evaluating candidates
        plf (pl.Fabric): lightning fabric instance

    Returns:
        th.Tensor: (n_tmpls, n_covs)
    """
    classifier.to(device=plf.device)
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    # NOTE init. candidate templates
    ctmpls: th.Tensor = (
        tafalib.makers.candidates.make_intra_correlation_template_candidates(
            xs=tdata["xs"],
            init_fidx=init_fidx,
            n_cands_targ=n_cands,
            min_features=min_features,
            max_features=max_features,
            corr_thrsh=corr_thrsh,
        )
    )
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(ctmpls)
    tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
        tmpls=ctmpls,
        data=(
            tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
            if max_tdata is not None and max_tdata < len(tdata)
            else tdata
        ),
        classifier=classifier,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )
    tmpls, slctd_ms = tafalib.makers.templates.make_templates_from_candidates(
        tpcomp=tpcomp,
        ctmpls=ctmpls,
        n_tmpls=n_tmpls,
        plf=plf,
        log_prefix="corr-mktmpl",
    )
    if (
        isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier)
        and to_update_classifier
    ):
        classifier.fit_(tmpls)
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
run_p: str = (
    "experiments/pretrain/nnet_subset_feature_classifier/outputs/big5/20250312_224514"
)
run_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, run_p, ".hydra", "config.yaml"))
tclassifier = (
    mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier.from_saved_state_dict(
        nnet=hd.utils.instantiate(
            run_cfg.nnet,
            in_features=n_covs * 2,
            out_features=n_labels,
        ),
        xs_train=tdata["xs"].numpy(),
        ys_train=tdata["ys"].numpy(),
        fit_kwargs=hd.utils.instantiate(run_cfg.nnet_fit_cfg),
        state_dict_p=os.path.join(PROJ_ROOT, run_p, "classifier.pt"),
    )
)
tclassifier.fit_kwargs["n_iter"] = 1000
tclassifier.fit_kwargs["bsz"] = 8192
vclassifier = tclassifier
max_tdata: Optional[int] = None
init_fidx: int = 35
n_tmpls_targ: int = 128
n_cands_targ: int = 10_000
lmbda: float = 0.075
min_features_targ: int = 1
max_features_targ: Optional[int] = None
bsz: int = 409600

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
output_dir: str = os.path.join("outputs", "run", data_name, "intracorr_cnnet")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")

# %%
if init_fidx is None:
    init_fidx, bestfm = tafalib.makers.templates.identify_init_fidx(
        tdata=tdata,
        classifier=tclassifier,
        max_features=max_features_targ,
        n_repeat=2,
        n_masks=500,
        lmbda=lmbda,
        bsz=bsz,
        plf=plf,
    )

# %%
tmpls = make_corr_templates(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    to_update_classifier=True,
    init_fidx=init_fidx,
    n_tmpls=n_tmpls_targ,
    n_cands=n_cands_targ,
    min_features=1,
    max_features=max_features_targ,
    corr_thrsh=0.3,
    lmbda=lmbda,
    bsz=bsz,
    plf=plf,
)

# %%
tpcomp: thd.TensorDict = tafalib.utils.precomp_rwds_for_tmpls(
    tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz, plf=plf
)

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=10,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    snfobsd_l.append(len(_fobsd_l))
    snfcomb_l.append(len(_fcomb))
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "init_fidx": init_fidx,
        "feature observed": th.mean(th.as_tensor(snfobsd_l, dtype=th.float32)).item(),
        "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
    }
)
print(pd.Series(metrics_d))


# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=10,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    snfobsd_l.append(len(_fobsd_l))
    snfcomb_l.append(len(_fcomb))
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "init_fidx": init_fidx,
        "feature used & observed": th.mean(
            th.as_tensor(snfobsd_l, dtype=th.float32)
        ).item(),
    }
)
print(pd.Series(metrics_d))

# %%
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = tafalib.utils.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=tclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=10,
            p=2,
        ),
        init_fidx=init_fidx,
        tmpls=tmpls,
        plf=plf,
    )
    snfobsd_l.append(len(_fobsd_l))
    snfcomb_l.append(len(_fcomb))
    metrics_func.update(
        _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
    )
metrics_d: dict[str, float] = {k: v.item() for k, v in metrics_func.compute().items()}
metrics_func.reset()
metrics_d.update(
    {
        "init_fidx": init_fidx,
        "feature used & observed": th.mean(
            th.as_tensor(snfobsd_l, dtype=th.float32)
        ).item(),
    }
)
print(pd.Series(metrics_d))

# %%
