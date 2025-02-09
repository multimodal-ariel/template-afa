from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Final, Generic, TypeVar

import mymodels.common
import mymodels.protocols
import numpy as np
import sklearn.cluster as skl_cluster
import sklearn.linear_model as skl_linear_model
import sklearn.neighbors as skl_neighbors
import torch as th
import xgboost as xgbst

MT = TypeVar("MT")
SKLMT = TypeVar("SKLMT", bound=mymodels.protocols.ModuleHasPredictProba)


class SubsetFeatureClassifier(th.nn.Module, ABC, Generic[MT]):
    # terminiology
    # act of tuple[int, ...] is an an indicator vector of length n_act_feats
    # fcomb of tuple[int, ...] is indices to selected subset of features (a.k.a feature combination)
    # exidx of int is index to which expert in case of n_experts_per_act is larget than 1

    n_experts_per_act: int
    xs_train: Final[np.ndarray]
    ys_train: Final[np.ndarray]

    n_covs: int

    _act_to_classifier: dict[tuple[int, ...], MT]

    _dummy: th.Tensor

    @property
    def device(self):
        return self._dummy.device

    def __init__(
        self,
        n_experts_per_act: int,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
    ):
        super().__init__()
        self.n_experts_per_act = n_experts_per_act
        self.xs_train = xs_train
        self.ys_train = ys_train
        # shape information
        self.n_covs = xs_train.shape[1]
        self.n_labels = len(np.unique(self.ys_train))
        # range of number of selected features
        # hparam for xgbc
        self._act_to_classifier = dict()
        # dummy to keep track of device
        self.register_buffer("_dummy", th.empty(()))

    # TODO this has been change; double check
    def act_to_fcomb_exidx(self, act: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        """transform an action to feature combination and expert index.

        Args:
            act (tuple[int, ...]): (n_act_feats, ) an action of interest

        Returns:
            tuple[int, ...]: feature combination
            int: expert index
        """
        if self.n_experts_per_act == 1:
            fcomb: tuple[int, ...] = tuple(
                th.argwhere(th.as_tensor(act, dtype=th.long) == 1).flatten().tolist()
            )
            return fcomb, 0
        fcomb: tuple[int, ...] = tuple(
            th.argwhere(th.as_tensor(act[: self.n_covs], dtype=th.long) == 1)
            .flatten()
            .tolist()
        )
        exidx: int = int(
            th.argwhere(th.as_tensor(act[self.n_covs :], dtype=th.long) == 1).item()
        )
        return fcomb, exidx

    @abstractmethod
    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        """Predict probability given context and action

        Args:
            ctxs (th.Tensor): (bsz, n_covs) the context
            acts (th.Tensor): (bsz, n_acts_covs) action taken for each context

        Returns:
            th.Tensor: (n_ctxs, n_labels) the probability
        """

    @abstractmethod
    def __getitem__(self, key: tuple[int, ...]) -> MT:
        """get classifier from xact

        Args:
            key (tuple[int, ...]): act in tuple, which is a vector of length n_act_feats

        Returns:
            MT: classifier
        """


class _SubsetFeatureSKLClassifier(SubsetFeatureClassifier[SKLMT]):
    make_model_func: Callable[[], SKLMT]
    use_cp: bool
    to_cache_model: bool

    def __init__(
        self,
        make_model_func: Callable[[], SKLMT],
        use_cp: bool,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        to_cache_model: bool,
    ):
        super().__init__(
            n_experts_per_act=1,
            xs_train=xs_train,
            ys_train=ys_train,
        )
        self.make_model_func = make_model_func
        self.use_cp = use_cp
        self.to_cache_model = to_cache_model

    # TODO this has been modified; double check
    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        n: int = len(ctxs)
        n_labels: int = self.n_labels
        acts = acts.to(dtype=th.long)
        acts_l: list[tuple[int, ...]] = [tuple(a.tolist()) for a in acts]
        pyhats: th.Tensor = th.empty((n, n_labels), dtype=th.float32)
        for act in set(acts_l):
            _act: th.Tensor = th.as_tensor([act], dtype=th.long, device=acts.device)
            _curr_idxs: th.Tensor = th.argwhere(th.all(acts == _act, dim=1)).flatten()
            _ctxs: th.Tensor = ctxs[_curr_idxs]
            pyhats[_curr_idxs] = self._predict_proba_same_act(_ctxs, act)
        return pyhats

    # TODO this has been modified; double check
    def _predict_proba_same_act(
        self, ctxs: th.Tensor, act: tuple[int, ...]
    ) -> th.Tensor:
        classifier = self[act]
        ctxs_ = (
            mymodels.common.to_cp_or_np(ctxs.to(device=self.device))
            if self.use_cp
            else ctxs.numpy(force=True)
        )
        fcomb: tuple[int, ...] = self.act_to_fcomb_exidx(act)[0]
        pyhats_n: np.ndarray = classifier.predict_proba(ctxs_[:, fcomb])
        pyhats: th.Tensor = th.as_tensor(pyhats_n, dtype=th.float32)
        return pyhats

    # TODO changed double check
    def __getitem__(self, key: tuple[int, ...]) -> SKLMT:
        assert self.n_experts_per_act == 1
        assert len(key) == self.n_covs
        if key in self._act_to_classifier:
            return self._act_to_classifier[key]
        act: tuple[int, ...] = key
        fcomb: tuple[int, ...] = self.act_to_fcomb_exidx(act)[0]
        model = self.make_model_func()
        xs = (
            mymodels.common.to_cp_or_np(
                th.as_tensor(
                    self.xs_train[:, fcomb], dtype=th.float32, device=self.device
                )
            )
            if self.use_cp
            else self.xs_train[:, fcomb].astype(np.float32)
        )
        ys = (
            mymodels.common.to_cp_or_np(
                th.as_tensor(self.ys_train, dtype=th.long, device=self.device)
            )
            if self.use_cp
            else self.ys_train.astype(np.int64)
        )
        model.fit(xs, ys)
        if self.to_cache_model:
            self._act_to_classifier[key] = model
        return model


class SubsetFeatureXGBClassifier(_SubsetFeatureSKLClassifier[xgbst.XGBClassifier]):
    xgbc_kwargs: dict[str, Any]

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        to_cache_model: bool = True,
        xgbc_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__(
            make_model_func=partial(xgbst.XGBClassifier, **xgbc_kwargs),
            use_cp=False,
            xs_train=xs_train,
            ys_train=ys_train,
            to_cache_model=to_cache_model,
        )
        # hparam for xgbc
        self.xgbc_kwargs = xgbc_kwargs


class SubsetFeatureLogisticRegressionClassifier(
    _SubsetFeatureSKLClassifier[skl_linear_model.LogisticRegression]  # type:ignore
):
    lrc_kwargs: dict[str, Any]

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        to_cache_model: bool = True,
        lrc_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__(
            make_model_func=self._make_model_func,
            use_cp=False,
            xs_train=xs_train,
            ys_train=ys_train,
            to_cache_model=to_cache_model,
        )
        self.lrc_kwargs = lrc_kwargs

    def _make_model_func(self) -> skl_linear_model.LogisticRegression:
        return skl_linear_model.LogisticRegression(**self.lrc_kwargs)


class SubsetFeatureKNNLogisticRegressionClassifier(SubsetFeatureClassifier[None]):
    knn_kwargs: dict[str, Any]
    lrc_kwargs: dict[str, Any]

    NearestNeighbors: Callable[..., skl_neighbors.NearestNeighbors]
    LogisticRegression: Callable[..., skl_linear_model.LogisticRegression]

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        knn_kwargs: dict[str, Any] = dict(),
        lrc_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__(
            n_experts_per_act=1,
            xs_train=xs_train,
            ys_train=ys_train,
        )
        self.knn_kwargs = knn_kwargs
        self.lrc_kwargs = lrc_kwargs
        self.NearestNeighbors = skl_neighbors.NearestNeighbors
        self.LogisticRegression = skl_linear_model.LogisticRegression

    # TODO changed double check
    def fit_predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[skl_linear_model.LogisticRegression],
        th.Tensor,
    ]:
        n: int = len(ctxs)
        n_labels: int = self.n_labels
        acts = acts.to(dtype=th.long)
        lrcs_l: list[skl_linear_model.LogisticRegression] = list()
        xs_train_lrc_l: list[np.ndarray] = list()
        ys_train_lrc_l: list[np.ndarray] = list()
        pyhats: th.Tensor = th.empty((n, n_labels), dtype=th.float32)
        for i, (ctx, act) in enumerate(zip(ctxs, acts)):
            _act: tuple[int, ...] = tuple(act.tolist())
            _comb, _ = self.act_to_fcomb_exidx(_act)
            _knn = self.NearestNeighbors(**self.knn_kwargs)
            _knn.fit(self.xs_train[:, _comb])
            _knnidxs: np.ndarray = _knn.kneighbors(
                ctx[None, _comb].numpy(force=True),
                return_distance=False,
            )  # type:ignore
            _knnidxs = _knnidxs.flatten()
            del _knn
            _lrc = self.LogisticRegression(**self.lrc_kwargs)
            _xs_train_lrc, _ys_train_lrc = self._make_valid_lrc_train_set(
                self.xs_train[_knnidxs][:, _comb], self.ys_train[_knnidxs]
            )
            _lrc.fit(_xs_train_lrc, _ys_train_lrc)
            pyhats[i] = th.as_tensor(
                _lrc.predict_proba(ctx[None, _comb].numpy(force=True)),
                dtype=th.float32,
            )
            lrcs_l.append(_lrc)
            xs_train_lrc_l.append(_xs_train_lrc)
            ys_train_lrc_l.append(_ys_train_lrc)
        return xs_train_lrc_l, ys_train_lrc_l, lrcs_l, pyhats

    # TODO changed double check
    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        n: int = len(ctxs)
        n_labels: int = self.n_labels
        pyhats: th.Tensor = th.empty((n, n_labels), dtype=th.float32)
        # batch nearest neighbors computations for inputs with same feature combinations
        acts_l: list[tuple[int, ...]] = [tuple(a.tolist()) for a in acts]
        for act in set(acts_l):
            # act: int = int(uact.item())
            fcomb: tuple[int, ...] = self.act_to_fcomb_exidx(act)[0]
            knn = self.NearestNeighbors(**self.knn_kwargs)
            knn.fit(self.xs_train[:, fcomb])
            # query knn in training set for input contexts with the same action
            _act: th.Tensor = th.as_tensor([act], dtype=th.long, device=acts.device)
            idxs: th.Tensor = th.argwhere(th.all(acts == _act, dim=1)).flatten()
            knnidxs: np.ndarray = knn.kneighbors(
                ctxs[idxs][:, fcomb].numpy(force=True), return_distance=False
            )  # type:ignore
            del knn
            # for each context, fit local personalized logistic regression model
            for _i, (_knnidxs, _ctx) in enumerate(zip(knnidxs, ctxs[idxs][:, fcomb])):
                _lrc = self.LogisticRegression(**self.lrc_kwargs)
                # ensure all labels shows up at least once
                _xs_train_lrc, _ys_train_lrc = self._make_valid_lrc_train_set(
                    self.xs_train[_knnidxs][:, fcomb], self.ys_train[_knnidxs]
                )
                _lrc.fit(_xs_train_lrc, _ys_train_lrc)
                _idx: int = int(idxs[_i].item())
                pyhats[_idx] = th.as_tensor(
                    _lrc.predict_proba(_ctx[None, :].numpy(force=True)),
                    dtype=th.float32,
                )
                del _lrc
        return pyhats

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None

    def _make_valid_lrc_train_set(
        self, xs_train: np.ndarray, ys_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_labels: int = self.n_labels
        if len(np.unique(ys_train)) == n_labels:
            return xs_train, ys_train
        dummy: np.ndarray = np.mean(xs_train, axis=0, keepdims=True)
        xs_train = np.concatenate((xs_train, np.tile(dummy, (n_labels, 1))), axis=0)
        ys_train = np.concatenate((ys_train, np.arange(n_labels, dtype=np.int64)))
        return xs_train, ys_train


class SubsetFeatureKNNClassifier(
    _SubsetFeatureSKLClassifier[skl_neighbors.KNeighborsClassifier]  # type:ignore
):
    knc_kwargs: dict[str, Any]

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        to_cache_model: bool = True,
        knc_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__(
            make_model_func=self._make_model_func,
            use_cp=False,
            xs_train=xs_train,
            ys_train=ys_train,
            to_cache_model=to_cache_model,
        )
        self.knc_kwargs = knc_kwargs

    def _make_model_func(self) -> skl_neighbors.KNeighborsClassifier:
        return skl_neighbors.KNeighborsClassifier(**self.knc_kwargs)


class SubsetFeatureNadarayaWatsonClassifier(SubsetFeatureClassifier[None]):
    _prior_exp: float
    _x_vars: np.ndarray

    def __init__(
        self,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        sig_mult: float = 0.15,
    ):
        super().__init__(
            n_experts_per_act=1,
            xs_train=xs_train,
            ys_train=ys_train,
        )
        assert len(np.unique(ys_train)) == 2
        self._prior_exp = np.mean(ys_train).item()
        self._x_vars = np.var(xs_train, axis=0, keepdims=True) * sig_mult

    # TODO
    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        acts = acts.to(dtype=th.long)
        xqrys: np.ndarray = ctxs.numpy(force=True)
        bs: np.ndarray = acts.numpy(force=True)
        return th.as_tensor(self._expert(xqrys, bs), dtype=th.float32)

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None

    def _nw_pred(
        self,
        Xtrn: th.Tensor,
        Ytrn: th.Tensor,
        Xq: th.Tensor,
        B: th.Tensor,
        sigmas: th.Tensor,
    ) -> th.Tensor:
        """
        Args:
            Xtrn: N x d Train Instances
            Ytrn: N x nclass Train Labels (one-hot)
            Xq: 1 x d Query Instances
            B: d x R binary masks to try
            sigmas: 1 x R bandwidth to use on each mask
        """
        Xtrn2 = Xtrn**2
        # Xq2 = Xq**2
        XtrnXq = Xtrn * Xq
        # N x R
        d2 = th.matmul(Xtrn2, B) - 2.0 * th.matmul(
            XtrnXq, B
        )  # TODO: don't think Xq2 needed
        kerns = th.softmax(-d2 / sigmas, dim=0)
        # R x nclass
        Y_neighbors = th.matmul(kerns.T, Ytrn)
        return Y_neighbors

    def _expert(
        self, X: np.ndarray, B: np.ndarray, alpha: float = 0.0, minprob: float = 0.001
    ) -> th.Tensor:
        """
        Implement some black box truth expert, should perform better when given
        relevent features, worse when not.

        Args:
            X: numpy array of shape (N, d) masked input features
            B: numpy array of shape (N, d) corresponding mask {0, 1}
        Returns:
            PY: numpy array of shape (N,) expert predicted probabilities
        """
        sig_masks = np.matmul(B, self._x_vars.T)
        sig_masks = np.maximum(
            sig_masks, np.min(self._x_vars)
        )  # Avoid div by zero for empty mask
        featspen = th.as_tensor(1.0 - alpha * np.mean(B, axis=1))
        # :( using slow for loop due to laziness
        # tho expert shouldn't be getting called too many times in practice
        PY = th.zeros((X.shape[0],), dtype=th.float32)
        for i in range(X.shape[0]):
            PY[i] = self._nw_pred(
                th.as_tensor(self.xs_train, dtype=th.float32, device=self.device),
                th.as_tensor(
                    self.ys_train[:, None], dtype=th.float32, device=self.device
                ),
                th.as_tensor(X[i, None, :], dtype=th.float32, device=self.device),
                th.as_tensor(B[i, None, :].T, dtype=th.float32, device=self.device),
                th.as_tensor(sig_masks[i], dtype=th.float32, device=self.device),
            ).to(device="cpu")
        PY = th.minimum(
            th.maximum(PY, th.as_tensor(minprob)), th.as_tensor(1.0 - minprob)
        )
        # TODO: adjust based on number of given feats
        # return featspen * PY + (1 - featspen) * self._prior_exp
        pyhats: th.Tensor = featspen * PY + (1 - featspen) * self._prior_exp
        pyhats = th.cat((1.0 - pyhats[:, None], pyhats[:, None]), dim=1)
        return pyhats


class SubsetFeatureMultiExpertNadarayaWatsonClassifier(SubsetFeatureClassifier[None]):
    _prior_exp: float
    _xs_train_multi: list[np.ndarray]
    _ys_train_multi: list[np.ndarray]
    _x_vars: np.ndarray

    def __init__(
        self,
        n_experts_per_act: int,
        xs_train: np.ndarray,
        ys_train: np.ndarray,
        kmeans_kwargs: dict[str, Any] = dict(),
        sig_mult: float = 0.15,
    ):
        super().__init__(
            n_experts_per_act=n_experts_per_act,
            xs_train=xs_train,
            ys_train=ys_train,
        )
        assert len(np.unique(ys_train)) == 2
        self._prior_exp = np.mean(ys_train).item()

        valid_clusters = False
        include_for_clustering = np.ones(xs_train.shape[0], dtype=bool)
        while not valid_clusters:

            cluster_model = skl_cluster.KMeans(
                n_clusters=n_experts_per_act, **kmeans_kwargs
            )
            cluster_model = cluster_model.fit(xs_train[include_for_clustering])
            cluster_labels: np.ndarray = cluster_model.predict(xs_train)

            valid_clusters = True
            for i in range(n_experts_per_act):
                if np.sum(cluster_labels == i) < 2:
                    valid_clusters = False
                    include_for_clustering *= cluster_labels != i

        self._xs_train_multi = list()
        self._ys_train_multi = list()
        self._x_vars = np.ones((n_experts_per_act, 1, self.n_covs), dtype=np.float32)
        for cid in np.unique(cluster_labels):
            _cms: np.ndarray = np.argwhere(cid == cluster_labels).flatten()
            self._xs_train_multi.append(xs_train[_cms])
            self._ys_train_multi.append(ys_train[_cms])
            self._x_vars[cid] = (
                np.var(self._xs_train_multi[cid], axis=0, keepdims=True) * sig_mult
                + 1e-5
            )
            assert np.all(self._x_vars[cid] > 0)

    # TODO changed; double check
    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        acts = acts.to(dtype=th.long)
        # TODO start here
        xqrys: np.ndarray = ctxs.numpy(force=True)
        bs: np.ndarray = acts.numpy(force=True)
        return th.as_tensor(self._expert(xqrys, bs), dtype=th.float32)

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None

    def _nw_pred(
        self,
        Xtrn: th.Tensor,
        Ytrn: th.Tensor,
        Xq: th.Tensor,
        B: th.Tensor,
        sigmas: th.Tensor,
    ) -> th.Tensor:
        """
        Args:
            Xtrn: N x d Train Instances
            Ytrn: N x nclass Train Labels (one-hot)
            Xq: 1 x d Query Instances
            B: d x R binary masks to try
            sigmas: 1 x R bandwidth to use on each mask
        """
        Xtrn2 = Xtrn**2
        # Xq2 = Xq**2
        XtrnXq = Xtrn * Xq
        # N x R
        d2 = th.matmul(Xtrn2, B) - 2.0 * th.matmul(
            XtrnXq, B
        )  # TODO: don't think Xq2 needed
        kerns = th.softmax(-d2 / sigmas, dim=0)
        # R x nclass
        Y_neighbors = th.matmul(kerns.T, Ytrn)
        return Y_neighbors

    def _expert(
        self, X: np.ndarray, B: np.ndarray, alpha: float = 0.0, minprob: float = 0.001
    ) -> th.Tensor:
        """
        Implement some black box truth expert, should perform better when given
        relevent features, worse when not.

        Args:
            X: numpy array of shape (N, d) masked input features
            B: numpy array of shape (N, d) corresponding mask {0, 1}
        Returns:
            PY: numpy array of shape (N,) expert predicted probabilities
        """
        expert_index: np.ndarray = np.argmax(B[:, self.n_covs :], axis=1)
        B_feat: np.ndarray = B[:, : self.n_covs]
        # (bsz, 1)
        sig_masks = np.take_along_axis(
            # (n_experts_per_fcomb, bsz, 1)
            np.stack(
                [
                    # (bsz, 1)
                    np.matmul(B_feat, self._x_vars[i].T)
                    for i in range(self.n_experts_per_act)
                ]
            ),
            # (bsz, 1)
            indices=expert_index[None, ..., None],
            axis=0,
        )[0]
        sig_masks = np.maximum(
            sig_masks, np.min(self._x_vars)
        )  # Avoid div by zero for empty mask
        featspen = th.as_tensor(1.0 - alpha * np.mean(B, axis=1))
        # :( using slow for loop due to laziness
        # tho expert shouldn't be getting called too many times in practice
        PY = th.zeros((X.shape[0],), dtype=th.float32)
        for i in range(X.shape[0]):
            PY[i] = self._nw_pred(
                th.as_tensor(
                    self._xs_train_multi[expert_index[i]],
                    dtype=th.float32,
                    device=self.device,
                ),
                th.as_tensor(
                    self._ys_train_multi[expert_index[i]][:, None],
                    dtype=th.float32,
                    device=self.device,
                ),
                th.as_tensor(X[i, None, :], dtype=th.float32, device=self.device),
                th.as_tensor(
                    B_feat[i, None, :].T, dtype=th.float32, device=self.device
                ),
                th.as_tensor(sig_masks[i], dtype=th.float32, device=self.device),
            ).to(device="cpu")
        PY = th.minimum(
            th.maximum(PY, th.as_tensor(minprob)), th.as_tensor(1.0 - minprob)
        )
        # TODO: adjust based on number of given feats
        pyhats: th.Tensor = featspen * PY + (1 - featspen) * self._prior_exp
        pyhats = th.cat((1.0 - pyhats[:, None], pyhats[:, None]), dim=1)
        return pyhats


class SubsetFeatureClassifierWrapperBase(SubsetFeatureClassifier[None], ABC):
    base_classifier: SubsetFeatureClassifier

    def __init__(self, base_classifier: SubsetFeatureClassifier):
        """SubsetFeatureClassifierBase

        Args:
            base_classifier (SubsetFeatureClassifier): base classifier
        """
        assert base_classifier.n_experts_per_act == 1
        super().__init__(
            n_experts_per_act=base_classifier.n_experts_per_act,
            xs_train=base_classifier.xs_train,
            ys_train=base_classifier.ys_train,
        )
        self.base_classifier = base_classifier

    def __getitem__(self, key: tuple[int, ...]) -> None:
        return None


class SubsetFeatureBiasedClassifierWrapperBase(SubsetFeatureClassifierWrapperBase):
    bias_level: float

    def __init__(self, base_classifier: SubsetFeatureClassifier, bias_level: float):
        """SubsetFeatureBiasedClassifierWrapperBase

        Args:
            base_classifier (SubsetFeatureClassifier[MT]): base classifier
            bias_level (float): strength of the bias effect within 0.0 and 1.0
        """
        super().__init__(base_classifier=base_classifier)
        assert base_classifier.n_labels == 2
        self.bias_level = bias_level

    @abstractmethod
    def _apply_bias(
        self, ctxs: th.Tensor, acts: th.Tensor, pyhats: th.Tensor
    ) -> th.Tensor: ...

    # TODO changed; double check
    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        acts = acts.to(dtype=th.long)
        pyhats: th.Tensor = self.base_classifier.predict_proba(ctxs, acts)
        assert pyhats.shape[1] == 2
        return self._apply_bias(ctxs, acts, pyhats)

    def _predict_proba_same_act(
        self, ctxs: th.Tensor, act: tuple[int, ...]
    ) -> th.Tensor:
        return NotImplemented


class SubsetFeatureOverloadClassifierWrapper(SubsetFeatureBiasedClassifierWrapperBase):
    min_temp: float
    bias_mult: float

    def __init__(
        self,
        base_classifier: SubsetFeatureClassifier,
        bias_level: float,
        min_temp: float = 1.0,
        bias_mult: float = 5.0,
    ):
        """SubsetFeatureOverloadClassifierWrapper
        Expert whose outputs increase in uncertainty as more features are provided in the selected display. This is implemented via a temperature function that increases with the square root of the proportion of observed features.

        Args:
            base_classifier (SubsetFeatureClassifier[MT]): base classifier to be wrapped
            bias_level (float): strength of the bias effect within 0.0 and 1.0
            min_temp (float, optional): temperature to apply to predictions when 0 features are provided
            bias_mult (float, optional): scale factor for bias level. Defaults to 5.0.
        """
        super().__init__(base_classifier, bias_level=bias_level)
        self.min_temp = min_temp
        self.bias_mult = bias_mult

    # TODO changed; double check
    def _apply_bias(
        self, ctxs: th.Tensor, acts: th.Tensor, pyhats: th.Tensor
    ) -> th.Tensor:
        # (bsz, 1)
        temp: th.Tensor = self.min_temp + self.bias_mult * self.bias_level * th.sqrt(
            th.mean(acts, dim=1, keepdim=True)
        )
        pyhats_: th.Tensor = pyhats ** (1 / temp)
        pyhats_ = pyhats_ / th.sum(pyhats_, dim=1, keepdim=True)
        return pyhats_


class SubsetFeatureRiskAverseClassifierWrapper(
    SubsetFeatureBiasedClassifierWrapperBase
):
    def __init__(self, base_classifier: SubsetFeatureClassifier, bias_level: float):
        """SubsetFeatureRiskAverseClassifierWrapper
        Expert with a bias towards predicting the positive class (Class 1).

        Here, a bias_level of 0 results in unchanged predictions, whereas a bias level of 1 will result in P(Y=1) = 1.0 in all cases.

        Args:
            base_classifier (SubsetFeatureClassifier[MT]): base classifier
            bias_level (float): strength of the bias effect within 0.0 and 1.0
        """
        super().__init__(base_classifier, bias_level=bias_level)

    def _apply_bias(
        self, ctxs: th.Tensor, acts: th.Tensor, pyhats: th.Tensor
    ) -> th.Tensor:
        pyhats_: th.Tensor = th.zeros_like(pyhats)
        pyhats_[:, 1] = (1 - self.bias_level) * pyhats[:, 1] + self.bias_level
        pyhats_[:, 0] = 1 - pyhats_[:, 1]
        return pyhats_


class SubsetFeaturePoisonFeatureClassifierWrapper(
    SubsetFeatureBiasedClassifierWrapperBase
):
    univar_model: xgbst.XGBClassifier
    poison_feature_idx: int

    def __init__(
        self,
        base_classifier: SubsetFeatureClassifier[MT],
        bias_level: float,
        poison_feature_idx: int,
        xgbc_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__(base_classifier, bias_level)
        self.poison_feature_idx = poison_feature_idx
        self.univar_model = xgbst.XGBClassifier(**xgbc_kwargs)
        self.univar_model.fit(
            base_classifier.xs_train[:, poison_feature_idx][:, None],
            base_classifier.ys_train,
        )

    def _apply_bias(
        self, ctxs: th.Tensor, acts: th.Tensor, pyhats: th.Tensor
    ) -> th.Tensor:
        acts_: th.Tensor = acts.to(dtype=th.bool)
        # (n_ctxs, )
        bias_levels: th.Tensor = th.where(
            acts_[:, self.poison_feature_idx], self.bias_level, 0.0
        )
        pyhats_u: th.Tensor = th.as_tensor(
            self.univar_model.predict_proba(
                ctxs[:, self.poison_feature_idx][:, None].numpy(force=True)
            ),
            dtype=th.float32,
            device=pyhats.device,
        )
        pyhats_: th.Tensor = th.zeros_like(pyhats)
        pyhats_[:, 1] = (1 - bias_levels) * pyhats[:, 1] + bias_levels * pyhats_u[:, 1]
        pyhats_[:, 0] = 1.0 - pyhats_[:, 1]
        return pyhats_
