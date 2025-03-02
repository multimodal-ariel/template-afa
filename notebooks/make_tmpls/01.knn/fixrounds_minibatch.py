# %%
from __future__ import annotations

import math
import os
from typing import Optional

import _classifiers
import _tmplfns
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mydatasets.aaco
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import pandas as pd
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm


# %%
def make_templates_fix_rounds_minibatch(
    tdata: thd.TensorDict,
    max_tdata: Optional[int],
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    n_tmpls_targ: int,
    n_cands_targ: int,
    n_cands_targ_minibatch: int,
    min_features: int,
    max_features: Optional[int],
    n_rounds: int,
    use_feature_importance_sampling: bool,
    lmbda: float,
    bsz: int,
    minibatch_size: int,
    plf: pl.Fabric,
) -> th.Tensor:
    n_covs: int = classifier.n_covs
    max_features = n_covs if max_features is None else max_features
    ctmpls: th.Tensor | None = None
    tmpls: th.Tensor | None = None
    slctd_ms: th.Tensor | None = None
    for _i in tqdm.trange(
        n_rounds, desc="mktmpl fix rounds", leave=False, dynamic_ncols=True
    ):
        if ctmpls is None or tmpls is None or slctd_ms is None:
            if isinstance(
                classifier, mymodels.classifiers.SubsetFeatureConcatClassifier
            ):
                classifier.fit_(
                    _tmplfns.make_template_candidates(
                        n_covs=n_covs,
                        init_fidx=init_fidx,
                        n_cands_targ=max(n_cands_targ, n_cands_targ_minibatch),
                        min_features=min_features,
                        max_features=max_features,
                    )
                )
            ctmpls = th.cat(
                [
                    _tmplfns.make_templates_vanilla(
                        tdata=tdata[
                            th.multinomial(
                                th.ones((len(tdata),)), num_samples=minibatch_size
                            )
                        ],
                        max_tdata=None,
                        classifier=classifier,
                        to_update_classifier=False,
                        init_fidx=init_fidx,
                        n_tmpls=n_tmpls_targ,
                        n_cands=n_cands_targ_minibatch,
                        min_features=min_features,
                        max_features=max_features,
                        lmbda=lmbda,
                        bsz=bsz,
                        plf=plf,
                    )
                    for _ in range(math.ceil(n_cands_targ / n_tmpls_targ))
                ],
                dim=0,
            )
        else:
            _ctmpls_prv: th.Tensor = ctmpls
            # update candidate pool from existing templates
            _ctmpls_l: list[th.Tensor] = [
                _tmplfns.update_template_candidates_fix_rounds(
                    ctmpls=_ctmpls_prv,
                    slctd_ms=slctd_ms,
                    init_fidx=init_fidx,
                    n_cands_init=n_cands_targ_minibatch,
                    n_cands_targ=n_cands_targ_minibatch,
                    min_features=min_features,
                    max_features=max_features,
                    use_feature_importance_sampling=use_feature_importance_sampling,
                )
                for _ in range(math.ceil(n_cands_targ / n_tmpls_targ))
            ]
            if isinstance(
                classifier, mymodels.classifiers.SubsetFeatureConcatClassifier
            ):
                classifier.fit_(th.unique(th.cat(_ctmpls_l, dim=0), dim=0))
            ctmpls = th.cat(
                [
                    _tmplfns.make_templates_from_candidates(
                        tpcomp=_tmplfns.precomp_rwds_for_tmpls(
                            tmpls=_ctmpls,
                            data=tdata[
                                th.multinomial(
                                    th.ones((len(tdata),)), num_samples=minibatch_size
                                )
                            ],
                            classifier=classifier,
                            lmbda=lmbda,
                            bsz=bsz,
                        ),
                        ctmpls=_ctmpls,
                        n_tmpls=n_tmpls_targ,
                        plf=plf,
                        log_prefix=f"round{_i}_minibatch{_j}",
                    )[0]
                    for _j, _ctmpls in enumerate(_ctmpls_l)
                ],
                dim=0,
            )
        ctmpls = th.unique(ctmpls, dim=0)
        assert ctmpls is not None
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(ctmpls)
        tpcomp: thd.TensorDict = _tmplfns.precomp_rwds_for_tmpls(
            ctmpls,
            data=(
                tdata[th.multinomial(th.ones((len(tdata),)), num_samples=max_tdata)]
                if max_tdata is not None and max_tdata < len(tdata)
                else tdata
            ),
            classifier=classifier,
            lmbda=lmbda,
            bsz=bsz,
        )
        tmpls, slctd_ms = _tmplfns.make_templates_from_candidates(
            tpcomp=tpcomp,
            ctmpls=ctmpls,
            n_tmpls=n_tmpls_targ,
            plf=plf,
            log_prefix=f"fixrounds_mktmpl{_i}",
        )
        if isinstance(classifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            classifier.fit_(tmpls)
    assert tmpls is not None
    return tmpls


# %%
# NOTE cube
# data_name: str = "cube_20_0.3"
# max_tdata: Optional[int] = None
# tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
# n_covs: int = tdata["xs"].shape[1]
# n_labels: int = len(th.unique(tdata["ys"]))
# classifier = SubsetFeatureNaiveBayes(
#     0.3, xs_train=tdata["xs"].numpy(), ys_train=tdata["ys"].numpy()
# )
# metrics_func = thm.MetricCollection(
#     {
#         "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
#         "precision": thm.Precision(task="multiclass", num_classes=n_labels),
#         "recall": thm.Recall(task="multiclass", num_classes=n_labels),
#         "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
#         "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
#     }
# )
# init_fidx: int = 6
# n_tmpls_targ: int = 64
# n_cands_targ: int = 10_000
# min_features_targ: int = 1
# max_features_targ: Optional[int] = None
# min_features_init: int = 10
# feature_decrement: int = 2
# lmbda: float = 0.3
# bsz: int = 1024

# %%
# NOTE big5
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
max_tdata: Optional[int] = 8192
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
tclassifier = _classifiers.SubsetFeatureConcatXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgb_kwargs={"n_estimators": 40},
    fraction_training_data_per_split=1.0,
    n_splits=64,
    n_tmpl_per_instance=4,
)
metrics_func = thm.MetricCollection(
    {
        "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
        "precision": thm.Precision(task="multiclass", num_classes=n_labels),
        "recall": thm.Recall(task="multiclass", num_classes=n_labels),
        "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
        "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
    }
)
init_fidx: int = 35
n_tmpls_targ: int = 128
n_cands_targ: int = 5_000
n_cands_targ_minibatch: int = 10_000
lmbda: float = 0.075
min_features_targ: int = 1
max_features_targ: Optional[int] = None
min_features_init: int = 10
minibatch_size: int = 64
n_rounds: int = 3
feature_decrement: int = 2
use_feature_importance_sampling: bool = True
bsz: int = 1024

# %%
# configure logger and ckpt path
output_dir: str = os.path.join("outputs", "run", data_name, "fixrounds_minibatch")
os.makedirs(output_dir, exist_ok=True)
tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="")
csv_logger = plf_loggers.CSVLogger(root_dir=tfb_logger.log_dir, name="", version="")

# %%
plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="cpu")

# %%
if init_fidx is None:
    init_fidx, bestfm = _tmplfns.identify_init_fidx(
        tdata=tdata,
        classifier=tclassifier,
        max_features=max_features_targ,
        n_repeat=2,
        n_iter=500,
        lmbda=lmbda,
        bsz=bsz,
    )


# %%
tmpls = make_templates_fix_rounds_minibatch(
    tdata=tdata,
    max_tdata=max_tdata,
    classifier=tclassifier,
    init_fidx=init_fidx,
    n_tmpls_targ=n_tmpls_targ,
    n_cands_targ=n_cands_targ,
    n_cands_targ_minibatch=n_cands_targ_minibatch,
    min_features=min_features_targ,
    max_features=max_features_targ,
    n_rounds=n_rounds,
    use_feature_importance_sampling=use_feature_importance_sampling,
    lmbda=lmbda,
    bsz=bsz,
    minibatch_size=minibatch_size,
    plf=plf,
)

# %%
tpcomp: thd.TensorDict = _tmplfns.precomp_rwds_for_tmpls(
    tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=lmbda, bsz=bsz
)

# %%
vclassifier = mymodels.classifiers.SubsetFeatureXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgbc_kwargs={"n_estimators": 40},
)
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: _tmplfns.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=2,
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
vclassifier = mymodels.classifiers.SubsetFeatureXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgbc_kwargs={"n_estimators": 40},
)
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=vclassifier,
        cost_est=lambda x: _tmplfns.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=2,
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
vclassifier = mymodels.classifiers.SubsetFeatureXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgbc_kwargs={"n_estimators": 40},
)
metrics_func.reset()
snfobsd_l: list[int] = list()
snfcomb_l: list[int] = list()
for _data in vdata:
    _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode_all_obsd(
        x=_data["xs"],
        classifier=tclassifier,
        cost_est=lambda x: _tmplfns.knn_cost_est(
            x,
            lmbda=lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=2,
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
