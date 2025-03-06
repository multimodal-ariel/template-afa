# %%
from __future__ import annotations

import os
import tempfile as tmpf
from typing import Any, Callable, Optional

import _tmplfns
import hydra as hd
import lightning as pl
import mymodels.classifiers
import numpy as np
import pandas as pd
import sklearn.exceptions as skl_exceptions
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst
from omegaconf import OmegaConf

# %%
PROJ_ROOT: str = "../../"

# %%
# mktmpl_run_dir: str = "experiments/make_template/outputs/grid/20250305_145621/2"
mktmpl_run_dir: str = "experiments/make_template/outputs/cube/20250305_143844/4"

# %%
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tdata_shuffle_idxs.pt")
)
tmpls: th.Tensor = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tmpls.pt"), weights_only=False
)
tpcomp: thd.TensorDict = th.load(
    os.path.join(PROJ_ROOT, mktmpl_run_dir, "tpcomp.pt"), weights_only=False
)
cfg = OmegaConf.load(os.path.join(PROJ_ROOT, mktmpl_run_dir, ".hydra", "config.yaml"))

# %%
# make dataset
_tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
_tdata, vdata, tstdata = hd.utils.call(cfg.data)
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))
# split training data into two for classifier and afa
tdata: thd.TensorDict
extdata: thd.TensorDict
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]

# %%
# make classifier
tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
    cfg.tclassifier, xs_train=extdata["xs"].numpy(), ys_train=extdata["ys"].numpy()
)
if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
    tclassifier.fit_(tmpls)
vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
if cfg.vclassifier is not None:
    vclassifier = hd.utils.instantiate(
        cfg.vclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )


# %%
class BootstrapXGBRegressor(th.nn.Module):
    # n_covs: int
    xgb_kwargs: dict[str, Any]
    # fraction_training_data_per_split: float
    n_splits: int

    _models: list[xgbst.XGBRegressor]
    _rg: th.Generator

    def __init__(
        self,
        # n_covs: int,
        xgb_kwargs: dict[str, Any],
        # fraction_training_data_per_split: float,
        n_splits: int,
        rseed: Optional[int] = None,
    ) -> None:
        super().__init__()
        super()
        # self.n_covs = n_covs
        self.xgb_kwargs = xgb_kwargs
        # self.fraction_training_data_per_split = fraction_training_data_per_split
        self.n_splits = n_splits
        self._models = [xgbst.XGBRegressor(**xgb_kwargs) for _ in range(n_splits)]
        self._rg = th.Generator()
        if rseed is not None:
            self._rg.manual_seed(rseed)
        # self.register_buffer("_dummy", th.empty(()))

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        _inputs: np.ndarray = inputs.numpy(force=True)
        outs_l: list[th.Tensor] = [
            th.as_tensor(m.predict(_inputs), device=inputs.device) for m in self._models
        ]
        # forward_outs (bsz, n_splits)
        outs: th.Tensor = th.stack(outs_l, dim=1)
        return outs

    def predict(self, inputs: th.Tensor) -> th.Tensor:
        outs: th.Tensor = th.mean(self.forward(inputs), dim=1)
        return outs

    def get_extra_state(self) -> Any:
        extra_state: dict[str, Any] = super().get_extra_state()
        model_states_l: list[list[str]] = list()
        extra_state.update(
            {
                "model_states_l": model_states_l,
                # "fraction_training_data_per_split": self.fraction_training_data_per_split,
                "n_splits": self.n_splits,
            }
        )
        try:
            with tmpf.TemporaryDirectory() as td:
                for i, model in enumerate(self._models):
                    p = os.path.join(td, f"m{i}.json")
                    model.save_model(p)
                    with open(p, mode="r") as f:
                        model_states: list[str] = f.readlines()
                        model_states_l.append(model_states)
        except skl_exceptions.NotFittedError:
            pass
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        super().set_extra_state(state)
        # self.fraction_training_data_per_split = state[
        #     "fraction_training_data_per_split"
        # ]
        self.n_splits = state["n_splits"]
        self._models.clear()
        with tmpf.TemporaryDirectory() as td:
            for i, model_states in enumerate(state["model_states_l"]):
                p = os.path.join(td, f"m{i}.json")
                with open(p, mode="w") as f:
                    f.writelines(model_states)
                model = xgbst.XGBRegressor()
                model.load_model(p)
                self._models.append(model)
        return


def bootstrap_xgb_regressor_cost_est(
    inps: th.Tensor, lmbda: float, tmpls: th.Tensor, regressor: BootstrapXGBRegressor
) -> th.Tensor:
    n_covs: int = tmpls.shape[1]
    device: th.device = inps.device
    # (n_tmpls, n_covs)
    tmpls = tmpls.to(device=device)
    # (bsz, n_tmpls, n_covs)
    fms_avail: th.Tensor = th.maximum(
        tmpls[None, :, :] - inps[:, None, n_covs:], th.as_tensor(0.0, device=device)
    )
    # (bsz, n_tmpls)
    cels: th.Tensor = regressor.predict(inps)
    costs: th.Tensor = cels + lmbda * th.sum(fms_avail, dim=2)
    return costs


# %%
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
def _make_warmup_fit_bsinps(
    bstdata: thd.TensorDict, init_fidx: int, tmpls: th.Tensor
) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bfms: th.Tensor = tmpls[btmplidxs].to(device=bxs.device)
    n_covs: int = bxs.shape[1]
    # randomly drop features
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs), device=bxs.device)
    bnms = th.clamp(bnms - bfms, 0.0, 1.0)
    bnms[:, init_fidx] = 1
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs * bnms, bnms), dim=1)
    return bsinps


def warmup_fit_xgb_regressor(
    regressor: BootstrapXGBRegressor,
    n_warmup_per_split: int,
    stdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
):
    pbar = tqdm.tqdm(
        regressor._models, desc="warmup fit", leave=False, dynamic_ncols=True
    )
    for m in pbar:
        bstdata: thd.TensorDict = stdata[
            th.randint(0, len(stdata), (n_warmup_per_split,))
        ]
        # (n_warmup_per_split, 2 * n_covs)
        bsinps: th.Tensor = _make_warmup_fit_bsinps(
            bstdata=bstdata, init_fidx=init_fidx, tmpls=tmpls
        )
        # (n_warmup_per_split, n_tmpls)
        bstargs: th.Tensor = bstdata["cels"]
        m.fit(bsinps.numpy(force=True), bstargs.numpy(force=True))


# %%
def _tmplafa_predict(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    n_labels: int = classifier.n_labels
    pyhats: th.Tensor = th.empty((len(data), n_labels), dtype=th.float32)
    oms: th.Tensor = th.zeros_like(data["xs"])
    fms: th.Tensor = th.zeros_like(data["xs"])
    for _i, _data in enumerate(data):
        _pyhat, _fobsd_l, _fcomb = _tmplfns.run_one_episode(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        pyhats[_i] = _pyhat
        oms[_i, _fcomb] = 1
        fms[_i, _fobsd_l] = 1
    return pyhats, oms, fms


def dagger_fit_xgb_reward_est(
    regressor: BootstrapXGBRegressor,
    n_data_per_split: int,
    stdata: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    init_fidx: int,
    lmbda: float,
    tmpls: th.Tensor,
    plf: pl.Fabric,
):
    pbar = tqdm.tqdm(
        regressor._models, desc="dagger fit", leave=False, dynamic_ncols=True
    )
    for m in pbar:
        bstdata: thd.TensorDict = stdata[
            th.randint(0, len(stdata), (n_data_per_split,))
        ]
        # (n_data_per_split, n_covs)
        _, boms, _ = _tmplafa_predict(
            data=bstdata,
            classifier=classifier,
            cost_est=lambda x: bootstrap_xgb_regressor_cost_est(
                x,
                lmbda=lmbda,
                tmpls=tmpls,
                regressor=regressor,
            ),
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        # (n_data_per_split, 2 * n_covs)
        bsinps: th.Tensor = th.cat((bstdata["xs"].to(device=plf.device), boms), dim=1)
        # (n_data_per_split, n_tmpls)
        bstargs: th.Tensor = bstdata["cels"]
        m.fit(
            bsinps.numpy(force=True),
            bstargs.numpy(force=True),
            xgb_model=m.get_booster(),
        )


# %%
stdata: thd.TensorDict = compile_selector_dataset(tdata=tdata, tpcomp=tpcomp)

# %%
regressor = BootstrapXGBRegressor({"n_estimators": 40}, n_splits=5)

# %%
warmup_fit_xgb_regressor(
    regressor=regressor,
    n_warmup_per_split=5 * len(stdata),
    stdata=stdata,
    init_fidx=cfg.init_fidx,
    tmpls=tmpls,
)

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
plf = pl.Fabric(accelerator="cpu")

# %%
metrics_func.reset()
metrics_d: dict[str, float] = _tmplfns.evaluate(
    data=vdata,
    classifier=vclassifier,
    cost_est=lambda x: bootstrap_xgb_regressor_cost_est(
        x, lmbda=cfg.lmbda, tmpls=tmpls, regressor=regressor
    ),
    init_fidx=cfg.init_fidx,
    tmpls=tmpls,
    metrics_func=metrics_func,
    plf=plf,
)
print(pd.Series(metrics_d))

# %%
dagger_fit_xgb_reward_est(
    regressor,
    n_data_per_split=5 * len(stdata),
    stdata=stdata,
    classifier=tclassifier,
    init_fidx=cfg.init_fidx,
    lmbda=cfg.lmbda,
    tmpls=tmpls,
    plf=plf,
)

# %%
