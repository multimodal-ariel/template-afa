from typing import Any, Protocol, runtime_checkable

import torch as th


@runtime_checkable
class ModuleHasForwardTrain(Protocol):
    def forward_train(
        self, xs: th.Tensor
    ) -> tuple[th.Tensor, dict[str, th.Tensor]]: ...


@runtime_checkable
class SklModule(Protocol):
    def fit(self, xs, ys, *args, **kwargs) -> Any: ...


@runtime_checkable
class ModuleHasPredict(SklModule, Protocol):
    def predict(self, xs, *args, **kwargs) -> Any: ...


@runtime_checkable
class ModuleHasPredictProba(ModuleHasPredict, Protocol):
    def predict_proba(self, xs, *args, **kwargs) -> Any: ...
