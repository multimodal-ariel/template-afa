# %%
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import aacolib
import dimelib
import hydra as hd
import jafalib
import lightning as pl
import mylib.utils
import mymodels
import numpy as np
import pytorch_lightning.plugins.environments as pl_plugins_envs
import tafalib
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

# %%
# big5
tafa_run_p: str = "experiments/make_template/outputs/big5_cnnet/20250318_144121/0"
aaco_run_p: str = "experiments/baselines/aco/outputs/big5/20250313_153149/0"
dime_run_p: str = "experiments/baselines/dime/outputs/big5_eval/20250319_144337/0"
jafa_run_p: str = "experiments/baselines/jafa/outputs/big5/20250326_231416/0"
n_instances: int = 10
# # cube
# tafa_run_p: str = "experiments/make_template/outputs/cube/20250318_225416/0"
# aaco_run_p: str = "experiments/baselines/aco/outputs/cube/20250311_201540/0"
# dime_run_p: str = "experiments/baselines/dime/outputs/cube_eval/20250323_223339/0"
# jafa_run_p: str = "experiments/baselines/jafa/outputs/cube/20250327_021409/0"
# n_instances: int = 10
# # gas
# tafa_run_p: str = "experiments/make_template/outputs/gas_cnnet/20250324_224734/16"
# aaco_run_p: str = "experiments/baselines/aco/outputs/gas/20250312_143952/0"
# dime_run_p: str = "experiments/baselines/dime/outputs/gas_eval/20250402_112555/0"
# jafa_run_p: str = "experiments/baselines/jafa/outputs/gas/20250327_125619/0"
# n_instances: int = 10
# # grid
# tafa_run_p: str = "experiments/make_template/outputs/grid_cnnet/20250325_213622/1"
# aaco_run_p: str = "experiments/baselines/aco/outputs/grid/20250311_221119/0"
# dime_run_p: str = "experiments/baselines/dime/outputs/grid_eval/20250323_220729/0"
# jafa_run_p: str = "experiments/baselines/jafa/outputs/grid/20250327_071151/0"
# n_instances: int = 10
# # mnist
# tafa_run_p: str = "experiments/make_template/outputs/mnist_cnnet/20250326_003820/1"
# aaco_run_p: str = "experiments/baselines/aco/outputs/mnist/20250326_163939/0"
# dime_run_p: str = "experiments/baselines/dime/outputs/mnist_eval/20250323_170930/0"
# jafa_run_p: str = "experiments/baselines/jafa/outputs/mnist/20250408_123011/0"
# n_instances: int = 10


# %%
def aaco_runtime():
    @dataclass
    class AACOMainConf:
        data: Any
        init_fidx: int
        lmbda: int
        n_neighs: int
        eval_bsz: int | None
        n_instances: int | None

    # Load the appropriate classifier based on dataset and model
    def load_classifier(dataset_name, X_train, y_train, input_dim):
        if dataset_name == "cube_20_0.3":
            # Use the ground truth classifier for Cube dataset
            return aacolib.classifier.classifier_ground_truth(
                num_features=20, num_classes=8, std=0.3
            )

        elif dataset_name == "grid_data" or dataset_name == "gas":
            # Use XGB dictionary classifier for Grid and Gas10 datasets
            return aacolib.classifier.classifier_xgb_dict(
                output_dim=len(th.unique(y_train)),
                input_dim=input_dim,
                subsample_ratio=0.01,
                X_train=X_train,
                y_train=y_train,
            )

        elif dataset_name == "mnist":
            # Load XGBoost model for MNIST dataset
            xgb_model = xgbst.XGBClassifier()
            xgb_model.load_model(
                os.path.join(
                    os.path.dirname(aacolib.__file__),
                    "_saved_models",
                    "mnist_xgb_classifier_arb_subset.json",
                )
            )
            return aacolib.classifier.classifier_xgb(xgb_model)
        elif dataset_name == "big5_C_cls":
            # Load XGBoost model for MNIST dataset
            xgb_model = xgbst.XGBClassifier()
            xgb_model.load_model(
                os.path.join(
                    os.path.dirname(aacolib.__file__),
                    "_saved_models",
                    "big5_xgb_classifier_arb_subsets.json",
                )
            )
            return aacolib.classifier.classifier_xgb(xgb_model)
        else:
            raise ValueError("Unsupported dataset or model")

    def get_knn(
        X_train, X_query, masks, num_neighbors, instance_idx=0, exclude_instance=True
    ):
        """
        Args:
        X_train: N x d Train Instances
        X_query: 1 x d Query Instances
        masks: d x R binary masks to try
        num_neighbors: Number of neighbors (k)
        """
        X_train_squared = X_train**2
        X_query_squared = X_query**2
        X_train_X_query = X_train * X_query
        dist_squared = (
            th.matmul(X_train_squared, masks)
            - 2.0 * th.matmul(X_train_X_query, masks)
            + th.matmul(X_query_squared, masks)
        )

        if exclude_instance:
            idx_topk = th.topk(dist_squared, num_neighbors + 1, dim=0, largest=False)[1]
            return idx_topk[idx_topk != instance_idx][:num_neighbors]
        else:
            return th.topk(dist_squared, num_neighbors, dim=0, largest=False)[1]

    # Helper function to load the mask generator based on the dataset
    def load_mask_generator(dataset_name, input_dim):
        if dataset_name in ["cube_20_0.3", "mnist", "big5_C_cls"]:
            return aacolib.mask_generator.random_mask_generator(10000, input_dim, 1000)
        elif dataset_name == "grid_data" or dataset_name == "gas":
            all_masks = aacolib.mask_generator.generate_all_masks(
                input_dim
            )  # Generate all possible masks for grid and gas10
            return aacolib.mask_generator.all_mask_generator(all_masks)
        else:
            raise ValueError("Unsupported dataset for mask generation")

    @th.no_grad()
    def aaco_rollout(
        X_train,
        y_train,
        X_valid,
        y_valid,
        classifier,
        mask_generator,
        initial_feature,
        acquisition_cost: float,
        n_neighs: int,
        is_train: bool = False,
        n_instances: Optional[int] = None,
    ):
        # Load parameters from the config
        feature_count = X_train.shape[1]
        # acquisition_cost = config["acquisition_cost"]
        # nearest_neighbors = config["nearest_neighbors"]
        acquisition_cost = acquisition_cost
        nearest_neighbors = n_neighs
        hide_val = 10
        # num_instances = config["num_instances"]  # Number of instances to loop through

        # Decide whether to use training or validation data
        # if config["train_or_validation"] == "train":
        if is_train:
            X = X_train
            y = y_train
            not_i = True  # Ensure instance isn't its own neighbor in KNN
        else:
            X = X_valid
            y = y_valid
            not_i = False  # Allow instance to be its own neighbor in KNN

        num_instances = n_instances if n_instances is not None else len(X)

        # Initialize lists to store results
        X_rollout = []
        y_rollout = []
        action_rollout = []
        mask_rollout = []

        # Define the loss function
        loss_function = th.nn.CrossEntropyLoss(reduction="none")

        ##############################################
        ##### AACO Rollout
        ##############################################
        # Loop through the specified number of instances
        for i in tqdm.trange(num_instances, dynamic_ncols=True, leave=False):
            # print(f"Starting instance {i} at {datetime.datetime.now()}")

            # Initialize the current mask (start with no features)
            mask_curr = th.zeros((1, feature_count))

            for j in range(feature_count + 1):
                if j == 0:
                    # Select the initial feature deterministically
                    mask_rollout.append(mask_curr.clone().detach())
                    mask_curr[0, initial_feature] = 1
                    action = th.zeros(1, feature_count + 1)
                    action[0, initial_feature] = 1
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    action_rollout.append(action)
                else:
                    # Get the nearest neighbors based on the observed feature mask
                    idx_nn = get_knn(
                        X_train, X[[i]], mask_curr.T, nearest_neighbors, i, not_i
                    ).squeeze()
                    # print(
                    #     f"Neighbors gathered for instance {i} at {datetime.datetime.now()}"
                    # )

                    # Generate random masks and get the next set of possible masks
                    new_masks = mask_generator(mask_curr)
                    mask = th.maximum(
                        new_masks, mask_curr.repeat(new_masks.shape[0], 1)
                    )
                    mask[0] = mask_curr  # Ensure the current mask is included

                    # Get only unique masks
                    mask = mask.unique(dim=0)
                    n_masks_updated = mask.shape[0]

                    # Predictions based on the classifier
                    x_rep = X_train[idx_nn].repeat(n_masks_updated, 1)
                    mask_rep = th.repeat_interleave(mask, nearest_neighbors, 0)
                    idx_nn_rep = idx_nn.repeat(n_masks_updated)
                    y_pred = classifier(
                        th.cat(
                            [
                                th.mul(x_rep, mask_rep) - (1 - mask_rep) * hide_val,
                                mask_rep,
                            ],
                            -1,
                        ),
                        idx_nn,
                    )

                    # Compute loss
                    loss = loss_function(
                        y_pred, y_train[idx_nn].repeat(n_masks_updated, 1).float()
                    ) + acquisition_cost * mask_rep.sum(dim=1)
                    loss = th.stack(
                        [
                            loss[
                                i * nearest_neighbors : (i + 1) * nearest_neighbors
                            ].mean()
                            for i in range(n_masks_updated)
                        ]
                    )

                    # Find the best mask (one with the lowest loss)
                    loss_argmin = loss.argmin()
                    mask_i = mask[loss_argmin]
                    mask_diff = mask_i - mask_curr

                    # Check if no new features are acquired
                    if mask_diff.sum().item() == 0:
                        # No more features to acquire, add prediction action
                        action = th.zeros(1, feature_count + 1)
                        action[0, feature_count] = (
                            1  # Action to predict (last column indicates prediction)
                        )
                        action_rollout.append(action)
                        X_rollout.append(X[[i]])
                        y_rollout.append(y[[i]])
                        mask_rollout.append(mask_curr)
                        break
                    else:
                        # If new features are acquired, choose the feature with the lowest expected loss
                        non_zero = mask_diff.nonzero()[:, 1]
                        ordering_masks = mask_curr.repeat(len(non_zero), 1)
                        ordering_masks[range(len(non_zero)), non_zero] = 1
                        ordering_masks = ordering_masks.repeat_interleave(
                            nearest_neighbors, 0
                        )

                        x_ordering = X_train[idx_nn].repeat(len(non_zero), 1)
                        idx_nn_ordering = idx_nn.repeat(len(non_zero))
                        y_pred = classifier(
                            th.cat(
                                [
                                    th.mul(x_ordering, ordering_masks)
                                    - (1 - ordering_masks) * hide_val,
                                    ordering_masks,
                                ],
                                -1,
                            ),
                            idx_nn,
                        )

                        # Compute loss for feature acquisition
                        loss = loss_function(
                            y_pred, y_train[idx_nn].repeat(len(non_zero), 1).float()
                        )
                        avg_loss = th.stack(
                            [
                                loss[
                                    i * nearest_neighbors : (i + 1) * nearest_neighbors
                                ].mean()
                                for i in range(len(non_zero))
                            ]
                        )

                        action_idx = non_zero[avg_loss.argmin()]
                        X_rollout.append(X[[i]])
                        y_rollout.append(y[[i]])
                        mask_rollout.append(mask_curr.clone().detach())
                        action = th.zeros(1, feature_count + 1)
                        action[0, action_idx] = 1
                        action_rollout.append(action)

                        # Update the current mask
                        mask_curr[:, action_idx] = 1
        # # Save the results
        #     results_dir = "./results/"
        #     os.makedirs(results_dir, exist_ok=True)

        #     data = {
        #         "X": th.cat(X_rollout),
        #         "mask": th.cat(mask_rollout),
        #         "Action": th.cat(action_rollout),
        #         "y": th.cat(y_rollout),
        #     }

        #     file_name = f"{results_dir}dataset_{config['dataset']}_rollout.pt"
        #     th.save(data, file_name)
        #     # print(f"Results saved to {file_name}")
        #     return data
        results: thd.TensorDict = thd.make_tensordict(
            {
                "X": th.cat(X_rollout),
                "mask": th.cat(mask_rollout),
                "Action": th.cat(action_rollout),
                "y": th.cat(y_rollout),
            }
        ).auto_batch_size_(batch_dims=1)
        return results

    aaco_cfg: AACOMainConf = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), aaco_run_p, ".hydra", "config.yaml"
        )
    )  # type:ignore
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(aaco_cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # load aaco prereq.
    classifier = load_classifier(
        dataset_name=aaco_cfg.data.name,
        X_train=tdata["xs"],
        y_train=tdata["ys"],
        input_dim=n_covs,
    )
    mask_generator = load_mask_generator(
        dataset_name=aaco_cfg.data.name, input_dim=n_covs
    )
    ts: th.Tensor = th.zeros((n_instances,), dtype=th.double)
    results: thd.TensorDict | None = None
    for _i in tqdm.trange(n_instances):
        _vidx: int = int(th.randint(0, len(vdata), ()).item())
        _start: float = time.time()
        _rst = aaco_rollout(
            X_train=tdata["xs"],
            y_train=th.nn.functional.one_hot(tdata["ys"], num_classes=n_labels),
            X_valid=vdata["xs"][_vidx : _vidx + 1],
            y_valid=th.nn.functional.one_hot(
                vdata["ys"][_vidx : _vidx + 1], num_classes=n_labels
            ),
            classifier=classifier,
            mask_generator=mask_generator,
            initial_feature=aaco_cfg.init_fidx,
            acquisition_cost=aaco_cfg.lmbda,
            n_neighs=aaco_cfg.n_neighs,
            is_train=False,
            n_instances=(
                aaco_cfg.n_instances if hasattr(aaco_cfg, "n_instances") else None
            ),
        )
        _end: float = time.time()
        ts[_i] = _end - _start
        if results is None:
            results = _rst
        else:
            results = thd.cat((results, _rst))
    time_avg: float = th.mean(ts).item()
    assert results is not None
    nfeats_avg: float = th.mean(
        th.sum(
            results["mask"][th.argwhere(results["Action"][:, -1] == 1).flatten(), :],
            dim=1,
        )
    ).item()
    print(f"time: {time_avg} n_feats: {nfeats_avg}")


print("aaco")
aaco_runtime()


# %%
def tafa_runtime():
    @dataclass
    class TAFAMainConf:
        data: Any
        tclassifier: Any
        vclassifier: Optional[Any]
        make_templates_fn: Any
        init_fidx: int
        lmbda: float
        n_neighs: int
        bsz: int
        plf: Any

    tafa_cfg: TAFAMainConf = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), tafa_run_p, ".hydra", "config.yaml"
        )
    )  # type:ignore
    _tdata_shuffle_idxs: th.Tensor = th.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), tafa_run_p, "tdata_shuffle_idxs.pt"
        )
    )
    tmpls: th.Tensor = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), tafa_run_p, "tmpls.pt"),
        weights_only=False,
    )
    tpcomp: thd.TensorDict = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), tafa_run_p, "tpcomp.pt"),
        weights_only=False,
    )
    # make dataset
    _tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    _tdata, vdata, tstdata = hd.utils.call(tafa_cfg.data)
    n_covs: int = _tdata["xs"].shape[1]
    n_labels: int = len(th.unique(_tdata["ys"]))
    # split training data into two for classifier and afa
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
    extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
    # make classifier
    tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
        tafa_cfg.tclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )
    if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        tclassifier_p = os.path.join(
            mylib.utils.get_project_root_dir(), tafa_run_p, "tclassifier.pt"
        )
        if os.path.exists(tclassifier_p):
            tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
        else:
            tclassifier.fit_(tmpls)
    vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
    if tafa_cfg.vclassifier is not None:
        vclassifier = hd.utils.instantiate(
            tafa_cfg.vclassifier,
            xs_train=extdata["xs"].numpy(),
            ys_train=extdata["ys"].numpy(),
        )
        if isinstance(vclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            vclassifier_p = os.path.join(
                mylib.utils.get_project_root_dir(), tafa_run_p, "tclassifier.pt"
            )
            if os.path.exists(vclassifier_p):
                vclassifier.load_state_dict(th.load(vclassifier_p, map_location="cpu"))
            else:
                vclassifier.fit_(tmpls)
    plf: pl.Fabric = hd.utils.instantiate(tafa_cfg.plf, _partial_=True)()
    _start: float = time.time()
    metrics_d: dict[str, float] = tafalib.utils.evaluate(
        data=vdata[th.multinomial(th.ones(len(vdata)), n_instances, replacement=False)],
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=tafa_cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=tafa_cfg.n_neighs,
            p=2,
        ),
        init_fidx=tafa_cfg.init_fidx,
        tmpls=tmpls,
        lmbda=tafa_cfg.lmbda,
        metrics_func=thm.MetricCollection(
            thm.Accuracy(task="multiclass", num_classes=n_labels)
        ),
        plf=plf,
    )
    _end: float = time.time()
    time_avg = (_end - _start) / n_instances
    nfeats_avg = metrics_d["feature observed"]
    print(f"time: {time_avg} n_feats: {nfeats_avg}")


print("tafa")
tafa_runtime()


# %%
def dime_runtime():
    def _get_run_dir(cfg) -> str:
        if hasattr(cfg, "train_run") and cfg.train_run is not None:
            return cfg.train_run
        assert hasattr(cfg, "train_exp") and cfg.train_exp is not None
        return os.path.join(cfg.train_exp.exp_p, str(cfg.train_exp.run_id))

    dime_ecfg = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), dime_run_p, ".hydra", "config.yaml"
        )
    )
    trun_p: str = _get_run_dir(dime_ecfg)
    dime_tcfg = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), trun_p, ".hydra", "config.yaml"
        )
    )
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(dime_tcfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    vdata = vdata[th.multinomial(th.ones(len(vdata)), n_instances, replacement=False)]
    vloader = th_data.DataLoader(
        th_data.TensorDataset(vdata["xs"], vdata["ys"]),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    # make nnet
    predictor_nnet: th.nn.Module = hd.utils.instantiate(
        dime_tcfg.predictor_nnet, in_features=n_covs * 2, out_features=n_labels
    )
    value_nnet: th.nn.Module = hd.utils.instantiate(
        dime_tcfg.value_nnet, in_features=n_covs * 2, out_features=n_covs
    )
    mask_layer = dimelib.utils.MaskLayer(mask_size=n_covs, append=True)
    if dime_tcfg.share_weights_layer_indices is not None:
        assert isinstance(predictor_nnet, th.nn.Sequential)
        assert isinstance(value_nnet, th.nn.Sequential)
        for _i in dime_tcfg.share_weights_layer_indices:
            value_nnet[_i] = predictor_nnet[_i]
    cmi_module_kwargs: dict[str, Any] = OmegaConf.to_container(
        dime_tcfg.cmi_module_cfg.cmi_module
    )  # type:ignore
    cmi_module_kwargs.pop("_target_")
    cmi_module = dimelib.cmi_estimator.CMIEstimator.load_from_checkpoint(
        os.path.join(
            mylib.utils.get_project_root_dir(),
            trun_p,
            "checkpoints",
            "best_val_perf_model.ckpt",
        ),
        value_network=value_nnet,
        predictor=predictor_nnet,
        mask_layer=mask_layer,
        **cmi_module_kwargs,
        loss_fn=th.nn.CrossEntropyLoss(reduction="none"),
        val_loss_fn=thm.Accuracy(task="multiclass", num_classes=n_labels),
        map_location="cpu",
    )
    # predict with validation
    trainer = pl.Trainer = hd.utils.instantiate(dime_ecfg.trainer, _partial_=True)(
        logger=False,
        plugins=[pl_plugins_envs.LightningEnvironment()],  # type:ignore
    )
    _start: float = time.time()
    inference_out: dict[str, th.Tensor] = cmi_module.inference(
        trainer, vloader, feature_costs=None, lam=dime_ecfg.lmbda
    )
    _end: float = time.time()
    time_avg = (_end - _start) / n_instances
    nfeats_avg = th.mean(th.sum(inference_out["mask"], dim=1)).item()
    print(f"time: {time_avg} n_feats: {nfeats_avg}")


print("dime")
dime_runtime()


# %%
def jafa_runtime():
    @dataclass
    class JAFAMainConf:
        disable_cuda: bool
        complete: bool
        pretrain: int
        pretrain_sample: str
        mode: str
        scheduler: str
        dropout: bool
        batchnorm: bool
        done_action_train: bool
        # data_type: str
        p: float
        group_norm: float
        save_dir: str
        embedder_hidden_sizes: list[int]
        clf_hidden_sizes: list[int]
        policy_hidden_sizes: list[int]
        shared_dim: int
        target_update_freq: int
        eps_start: float
        eps_end: float
        decay_rate: float
        n_envs: int
        nsteps: int
        normalize: bool
        embedded_dim: int
        lstm_size: int
        n_shuffle: int
        r_cost: float
        cost_from_file: bool
        random_seed: int
        batch_size: int
        message: str
        # post processing
        device: th.device
        save_dir: str
        save_path: str
        csv_path: str

    def make_default_jafa_cfg() -> JAFAMainConf:
        args = JAFAMainConf(
            disable_cuda=False,
            complete=False,
            pretrain=10_000,
            pretrain_sample="both",
            mode="double",
            scheduler="linear",
            dropout=False,
            batchnorm=False,
            done_action_train=False,
            p=0,
            group_norm=0.0,
            save_dir="",
            embedder_hidden_sizes=[32, 32],
            clf_hidden_sizes=[32, 32],
            policy_hidden_sizes=[32],
            shared_dim=16,
            target_update_freq=100,
            eps_start=1.0,
            eps_end=0.1,
            decay_rate=2.0,
            n_envs=128,
            nsteps=4,
            normalize=True,
            embedded_dim=16,
            lstm_size=16,
            n_shuffle=5,
            r_cost=-0.05,
            cost_from_file=False,
            random_seed=123,
            batch_size=128,
            message="",
            device=th.device("cpu"),
            save_path="",
            csv_path="",
        )
        return args

    def override_jafa_cfg_(
        jafa_args: JAFAMainConf, jafa_cfg: Any, output_dir: str, plf: pl.Fabric
    ):
        # copy from hydra config to jafa args
        for _k in OmegaConf.to_container(jafa_cfg).keys():
            assert hasattr(jafa_args, _k), f"{_k} is an invalid jafa argument"
        for k in asdict(jafa_args):
            if hasattr(jafa_cfg, k):
                setattr(jafa_args, k, getattr(jafa_cfg, k))
        if plf.device.type == "cuda":
            jafa_args.device = plf.device
            jafa_args.disable_cuda = False
        jafa_args.save_dir = output_dir
        jafa_args.save_path = os.path.join(jafa_args.save_dir, "outputs")
        os.makedirs(jafa_args.save_path, exist_ok=True)
        jafa_args.csv_path = os.path.join(jafa_args.save_dir, "jafa_results")

    def to_jafa_datatmp(
        data: thd.TensorDict, shuffle: bool
    ) -> jafalib.data_temp.DataTemp:
        xs: th.Tensor = data["xs"]
        ys: th.Tensor = data["ys"]
        n_labels: int = len(th.unique(ys))
        return jafalib.data_temp.DataTemp(
            xs.numpy(force=True),
            ys.numpy(force=True),
            exist=None,
            n_classes=n_labels,
            shuffle=shuffle,
            # iter=iter,
            action2features=None,
        )

    def test_and_record(
        step_runner: jafalib.main.StepRunner,
        args,
        env: jafalib.environment.Env,
        valenv: Optional[jafalib.environment.Env] = None,
        testenv: Optional[jafalib.environment.Env] = None,
    ):
        print("=================== Test start ==================")
        fieldnames = [
            "acc",
            "n_acquired_mean",
            "n_acquired_min",
            "n_acquired_max",
            "n_acquired_med",
            "return",
        ]
        n_features, n_classes = env.n_features, env.n_classes
        if n_classes == 2:
            fieldnames = ["auc"] + fieldnames
        fieldnames += ["picked_{}".format(i) for i in range(n_features)]
        argsdict = dict(vars(args))
        argsdict.pop("message")
        sorted_argskey = sorted(argsdict.keys())
        field = []
        result = {}
        tstresults_l = dict()
        for prefix, environ in [("val", valenv), ("tr", env), ("ts", testenv)]:
            step_runner.load(os.path.join(args.save_path, "trained_best.model"))
            if environ is None:
                break
            environ.reset()
            test_result = jafalib.main.test(step_runner, environ, args)
            tstresults_l[prefix] = test_result
            correct = test_result[1]
            acquired = test_result[2]
            returns = test_result[3]
            field += [prefix + "_" + field for field in fieldnames]
            result[prefix + "_acc"] = np.mean(correct)
            result[prefix + "_n_acquired_mean"] = np.mean(np.sum(acquired, 1))
            result[prefix + "_n_acquired_min"] = np.amin(np.sum(acquired, 1))
            result[prefix + "_n_acquired_max"] = np.amax(np.sum(acquired, 1))
            result[prefix + "_n_acquired_med"] = np.median(np.sum(acquired, 1))
            if n_classes == 2:
                result[prefix + "_auc"] = test_result[-1]  # auc
                test_result = test_result[:-3]
            for i in range(n_features):
                result[prefix + "_picked_{}".format(i)] = np.sum(acquired[:, i])
            result[prefix + "_return"] = np.mean(returns)
        field = sorted_argskey + field
        result.update(argsdict)

    jafa_cfg = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), jafa_run_p, ".hydra", "config.yaml"
        )
    )
    # instantiate dataset
    tdata, vdata, tstdata = hd.utils.instantiate(jafa_cfg.data)
    tjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(tdata, False)
    vjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(
        vdata[th.multinomial(th.ones(len(vdata)), n_instances, replacement=False)],
        False,
    )
    # configure jafa configs
    plf: pl.Fabric = hd.utils.instantiate(jafa_cfg.plf, _partial_=True)()
    jafa_args: JAFAMainConf = make_default_jafa_cfg()
    override_jafa_cfg_(
        jafa_args=jafa_args, jafa_cfg=jafa_cfg.jafa_cfg, output_dir="", plf=plf
    )
    jafa_args.n_envs = n_instances
    jafa_args.batch_size = 1
    jafa_args.save_path = os.path.join(
        mylib.utils.get_project_root_dir(), jafa_run_p, "outputs"
    )
    # run main function of jafa
    embedder_hidden_sizes = jafa_args.embedder_hidden_sizes
    embedded_dim = jafa_args.embedded_dim
    lstm_size = jafa_args.lstm_size
    n_shuffle = jafa_args.n_shuffle
    clf_hidden_sizes = jafa_args.clf_hidden_sizes
    policy_hidden_sizes = jafa_args.policy_hidden_sizes
    shared_dim = jafa_args.shared_dim
    nsteps = jafa_args.nsteps
    n_envs = jafa_args.n_envs
    r_cost = jafa_args.r_cost
    cost = None
    if cost is not None:
        r_cost = cost
    input_dim = tjdatatmp.n_features + 1
    clf_output_size = tjdatatmp.n_classes if tjdatatmp.n_classes > 2 else 1
    encoder = jafalib.model.SetEncoder(
        input_dim,
        tjdatatmp.n_features,
        embedder_hidden_sizes,
        embedded_dim,
        lstm_size,
        n_shuffle,
        normalize=jafa_args.normalize,
        dropout=jafa_args.dropout,
        p=jafa_args.p,
    )
    dfsnet = jafalib.model.DFSNet(
        encoder=encoder,
        classifier=jafalib.model.MLP(
            lstm_size + embedded_dim,
            clf_hidden_sizes,
            clf_output_size,
            dropout=jafa_args.dropout,
            p=jafa_args.p,
            batch_norm=jafa_args.batchnorm,
        ),
        policy=jafalib.model.DuelingNet(
            lstm_size + embedded_dim,
            policy_hidden_sizes,
            shared_dim,
            tjdatatmp.n_actions,
        ),
    )
    dfsnet.eval().to(device=plf.device)
    step_runner = jafalib.main.StepRunner(dfsnet, jafa_args)
    step_runner.load(
        os.path.join(
            mylib.utils.get_project_root_dir(),
            jafa_run_p,
            "outputs",
            "trained_best.model",
        )
    )
    valenv = jafalib.environment.Env(
        jafa_args,
        n_envs,
        r_cost,
        vjdatatmp,
        step_runner.classify,
    )
    valenv.classify = step_runner.classify
    dfsnet.eval()
    _start: float = time.time()
    test_and_record(step_runner, jafa_args, valenv, None, None)
    _end: float = time.time()
    time_avg = (_end - _start) / n_instances
    print(f"time: {time_avg}")


print("jafa")
jafa_runtime()


# %%
