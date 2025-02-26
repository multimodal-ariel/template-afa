from __future__ import annotations

import math
import os
import tempfile as tmpf
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import lightning as pl
import numpy as np
import sklearn.exceptions as skl_exceptions
import tensordict as thd
import torch as th
import xgboost as xgbst

OutputT = TypeVar("OutputT")


class RewardEst(th.nn.Module, ABC, Generic[OutputT]):
    n_ctx_covs: int
    _train_inputs: th.Tensor | None
    _train_targets: th.Tensor | None
    _infos: th.Tensor | thd.TensorDictBase | None

    _dummy: th.Tensor

    @property
    def device(self):
        return self._dummy.device

    @property
    def train_inputs(self) -> th.Tensor:
        assert self._train_inputs is not None
        return self._train_inputs

    @property
    def train_targets(self) -> th.Tensor:
        assert self._train_targets is not None
        return self._train_targets

    def __init__(self, n_ctx_covs: int) -> None:
        super().__init__()
        self.n_ctx_covs = n_ctx_covs
        self._train_inputs = None
        self._train_targets = None
        self._infos = None
        self.register_buffer("_dummy", th.empty(()))

    def set_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: Optional[th.Tensor | thd.TensorDictBase] = None,
    ):
        assert len(inputs) == len(targets.flatten())
        self._train_inputs = inputs.clone().to(device="cpu")
        self._train_targets = targets.clone().flatten().to(device="cpu")
        if infos is not None:
            self._infos = infos.clone().to(device="cpu")

    def add_to_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: Optional[th.Tensor | thd.TensorDictBase] = None,
    ):
        new_inputs = th.cat((self.train_inputs, inputs.to(device="cpu")), dim=0)
        new_targets = th.cat((self.train_targets, targets.to(device="cpu")), dim=0)
        if infos is not None and self._infos is not None:
            infos = thd.cat((self._infos, infos.to(device="cpu")), dim=0)
        self.set_train_data_(new_inputs, new_targets, infos)

    def initialize(self, xs: th.Tensor, ys: th.Tensor, *args, **kwargs):
        pass

    @abstractmethod
    def decompose_inputs(
        self, inputs: th.Tensor
    ) -> tuple[th.Tensor, thd.TensorDictBase]:
        """decompose inputs into ctxs, cinds, exinds

        Args:
            inputs (th.Tensor): (bsz, n_covs)

        Returns:
            th.Tensor: (bsz, n_ctx_covs) contexts
            thd.TensorDictBase: (bsz, ) actions
        """

    @abstractmethod
    def forward(self, inputs: th.Tensor) -> OutputT: ...

    @abstractmethod
    def get_posterior_mean(self, forward_outs: OutputT) -> th.Tensor: ...

    @abstractmethod
    def get_posterior_covariance(self, forward_outs: OutputT) -> th.Tensor: ...

    @abstractmethod
    def sample_posterior(
        self, forward_outs: OutputT, sample_shape: th.Size = th.Size()
    ) -> th.Tensor: ...

    @abstractmethod
    def fit_(self, plf: pl.Fabric) -> dict[str, float]: ...

    def get_extra_state(self) -> dict[str, Any]:
        extra_state: dict[str, Any] = {
            "train_inputs": self._train_inputs,
            "train_targets": self._train_targets,
            "infos": self._infos,
        }
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        self._train_inputs = state["train_inputs"]
        self._train_targets = state["train_targets"]
        self._infos = state["infos"]


class RewardEstWrapperBase(RewardEst[dict[str, Any]], ABC, Generic[OutputT]):
    base_est: RewardEst[OutputT]

    def __init__(self, base_est: RewardEst[OutputT]) -> None:
        super().__init__(base_est.n_ctx_covs)
        self.base_est = base_est

    def forward(self, inputs: th.Tensor) -> dict[str, Any]:
        base_forward_outs: OutputT = self.base_est(inputs)
        outputs: dict[str, Any] = {
            "base_forward_outs": base_forward_outs,
        }
        return outputs

    def get_posterior_mean(self, forward_outs: dict[str, Any]) -> th.Tensor:
        return self.base_est.get_posterior_mean(forward_outs["base_forward_outs"])

    def get_posterior_covariance(self, forward_outs: dict[str, Any]) -> th.Tensor:
        return self.base_est.get_posterior_covariance(forward_outs["base_forward_outs"])

    def sample_posterior(
        self, forward_outs: dict[str, Any], sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        return self.base_est.sample_posterior(forward_outs["base_forward_outs"])

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        metrics_d = self.base_est.fit_(plf)
        return metrics_d

    @property
    def device(self):
        return self.base_est.device

    @property
    def train_inputs(self) -> th.Tensor:
        return self.base_est.train_inputs

    @property
    def train_targets(self) -> th.Tensor:
        return self.base_est.train_targets

    def set_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: th.Tensor | thd.TensorDictBase | None = None,
    ):
        return self.base_est.set_train_data_(inputs, targets, infos)

    def add_to_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: th.Tensor | thd.TensorDictBase | None = None,
    ):
        return self.base_est.add_to_train_data_(inputs, targets, infos)

    def initialize(self, xs: th.Tensor, ys: th.Tensor, *args, **kwargs):
        return self.base_est.initialize(xs, ys)

    def decompose_inputs(
        self, inputs: th.Tensor
    ) -> tuple[th.Tensor, thd.TensorDictBase]:
        return self.base_est.decompose_inputs(inputs)

    def get_extra_state(self) -> dict[str, Any]:
        return self.base_est.get_extra_state()

    def set_extra_state(self, state: Any) -> None:
        return self.base_est.set_extra_state(state)


class _XGBRewardEstBase(RewardEst[th.Tensor]):
    xgb_kwargs: dict[str, Any]
    fraction_training_data_per_split: float
    n_splits: int

    _models: list[xgbst.XGBRegressor]
    _rg: th.Generator

    def __init__(
        self,
        xgb_kwargs: dict[str, Any],
        models: list[xgbst.XGBModel],
        n_ctx_covs: int,
        fraction_training_data_per_split: float,
        n_splits: int,
        rseed: Optional[int] = None,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs)
        self.xgb_kwargs = xgb_kwargs
        self.fraction_training_data_per_split = fraction_training_data_per_split
        self.n_splits = n_splits
        self._models = models
        self._rg = th.Generator()
        if rseed is not None:
            self._rg.manual_seed(rseed)

    def decompose_inputs(
        self, inputs: th.Tensor
    ) -> tuple[th.Tensor, thd.TensorDictBase]:
        return super().decompose_inputs(inputs)

    def get_posterior_mean(self, forward_outs: th.Tensor) -> th.Tensor:
        # forward_outs (bsz, n_splits)
        return th.mean(forward_outs, dim=1)

    def get_posterior_covariance(self, forward_outs: th.Tensor) -> th.Tensor:
        assert self.n_splits > 1
        # forward_outs (bsz, n_splits)
        mus: th.Tensor = th.mean(forward_outs, dim=1, keepdim=True)
        fouts_minus_mus: th.Tensor = forward_outs - mus
        # (bsz, bsz)
        covars: th.Tensor = (1 / (self.n_splits - 1)) * (
            fouts_minus_mus @ fouts_minus_mus.T
        )
        covars = th.nan_to_num(covars, nan=0, posinf=999, neginf=999)
        return covars

    def sample_posterior(
        self, forward_outs: th.Tensor, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        n: int = len(forward_outs)
        n_samps: int = math.prod(sample_shape)
        idxs: th.Tensor = th.randint(
            0, len(self._models), (n, n_samps), dtype=th.long, device=self.device
        )
        outs: th.Tensor = th.gather(forward_outs, dim=1, index=idxs).reshape(
            (n, *sample_shape)
        )
        return outs

    def get_extra_state(self) -> Any:
        extra_state: dict[str, Any] = super().get_extra_state()
        model_states_l: list[list[str]] = list()
        extra_state.update(
            {
                "model_states_l": model_states_l,
                "fraction_training_data_per_split": self.fraction_training_data_per_split,
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

    @abstractmethod
    def set_extra_state(self, state: Any) -> None:
        super().set_extra_state(state)
        self.fraction_training_data_per_split = state[
            "fraction_training_data_per_split"
        ]
        self.n_splits = state["n_splits"]
        self._models.clear()
        return


class XGBoostRewardEst(_XGBRewardEstBase):
    xgbr_kwargs: dict[str, Any]

    def __init__(
        self,
        n_ctx_covs: int,
        fraction_training_data_per_split: float,
        n_splits: int,
        xgbr_kwargs: dict[str, Any] = dict(),
        rseed: Optional[int] = None,
    ) -> None:
        super().__init__(
            xgb_kwargs=xgbr_kwargs,
            models=[xgbst.XGBRegressor(**xgbr_kwargs) for _ in range(n_splits)],
            n_ctx_covs=n_ctx_covs,
            fraction_training_data_per_split=fraction_training_data_per_split,
            n_splits=n_splits,
            rseed=rseed,
        )
        self.xgbr_kwargs = xgbr_kwargs

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        _inputs: np.ndarray = inputs.numpy(force=True)
        outs_l: list[th.Tensor] = [
            th.as_tensor(m.predict(_inputs), device=self.device) for m in self._models
        ]
        outs: th.Tensor = th.stack(outs_l, dim=1)
        return outs

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        self.train()
        rsquares_l: list[float] = list()
        mses_l: list[float] = list()
        for m in self._models:
            n_data: int = math.ceil(
                len(self.train_targets) * self.fraction_training_data_per_split
            )
            idxs: th.Tensor = th.randint(
                0, len(self.train_targets), (n_data,), dtype=th.long, generator=self._rg
            )
            xs: np.ndarray = self.train_inputs[idxs,].numpy(force=True)
            ys: np.ndarray = self.train_targets[idxs].numpy(force=True)
            m.fit(xs, ys)
            train_inputs: np.ndarray = self.train_inputs.numpy(force=True)
            train_targets: np.ndarray = self.train_targets.numpy(force=True)
            rsquares_l.append(m.score(train_inputs, train_targets))
            mses_l.append(
                th.nn.functional.mse_loss(
                    th.as_tensor(m.predict(train_inputs), dtype=th.float32),
                    self.train_targets,
                ).item()
            )
        metrics = {
            "est_rsquared": th.mean(th.as_tensor(rsquares_l)).item(),
            "est_mse": th.mean(th.as_tensor(mses_l)).item(),
        }
        return metrics

    def set_extra_state(self, state: Any) -> None:
        super().set_extra_state(state)
        with tmpf.TemporaryDirectory() as td:
            for i, model_states in enumerate(state["model_states_l"]):
                p = os.path.join(td, f"m{i}.json")
                with open(p, mode="w") as f:
                    f.writelines(model_states)
                model = xgbst.XGBRegressor()
                model.load_model(p)
                self._models.append(model)
        return
