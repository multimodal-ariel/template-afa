from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import aacolib.classifier
import aacolib.mask_generator
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib.utils
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm.auto as tqdm
import xgboost as xgbst
from hydra.core.hydra_config import HydraConfig


@dataclass
class MainConf:
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
    elif dataset_name == "volvo":
        xgb_model = xgbst.XGBClassifier()
        xgb_model.load_model(
            os.path.join(
                os.path.dirname(aacolib.__file__),
                "_saved_models",
                "volvo_xgb_classifier_arb_subsets.json",
            )
        )
        return aacolib.classifier.classifier_xgb(xgb_model)
    elif dataset_name == "charfont-1500":
        xgb_model = xgbst.XGBClassifier()
        xgb_model.load_model(
            os.path.join(
                os.path.dirname(aacolib.__file__),
                "_saved_models",
                "charfont1500_xgb_classifier_arb_subsets.json",
            )
        )
        return aacolib.classifier.classifier_xgb(xgb_model)
    elif dataset_name == "ct-slice-rel-loc":
        xgb_model = xgbst.XGBClassifier()
        xgb_model.load_model(
            os.path.join(
                os.path.dirname(aacolib.__file__),
                "_saved_models",
                "ct-slice-rel-loc_xgb_classifier_arb_subsets.json",
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
    if dataset_name in [
        "cube_20_0.3",
        "mnist",
        "big5_C_cls",
        "volvo",
        "charfont-1500",
        "ct-slice-rel-loc",
    ]:
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
                mask = th.maximum(new_masks, mask_curr.repeat(new_masks.shape[0], 1))
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
                        loss[i * nearest_neighbors : (i + 1) * nearest_neighbors].mean()
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


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # load aaco prereq.
    classifier = load_classifier(
        dataset_name=cfg.data.name,
        X_train=tdata["xs"],
        y_train=tdata["ys"],
        input_dim=n_covs,
    )
    mask_generator = load_mask_generator(dataset_name=cfg.data.name, input_dim=n_covs)
    # run aaco
    results: thd.TensorDict = aaco_rollout(
        X_train=tdata["xs"],
        y_train=th.nn.functional.one_hot(tdata["ys"], num_classes=n_labels),
        X_valid=vdata["xs"],
        y_valid=th.nn.functional.one_hot(vdata["ys"], num_classes=n_labels),
        classifier=classifier,
        mask_generator=mask_generator,
        initial_feature=cfg.init_fidx,
        acquisition_cost=cfg.lmbda,
        n_neighs=cfg.n_neighs,
        is_train=False,
        n_instances=cfg.n_instances if hasattr(cfg, "n_instances") else None,
    )
    th.save(results, os.path.join(output_dir, "results.pt"))
    # forward prop. classifier
    ridxs: th.Tensor = th.argwhere(results["Action"][:, -1] == 1).flatten()
    # # without batch
    # inps = th.cat(
    #     [
    #         result["X"][ridxs] * result["mask"][ridxs]
    #         - (1 - result["mask"][ridxs]) * 10,
    #         result["mask"][ridxs],
    #     ],
    #     dim=1,
    # )
    # pyhats: th.Tensor = classifier(inps, None)
    # eval split with batch
    bsz: int = cfg.eval_bsz if cfg.eval_bsz is not None else len(ridxs)
    pyhats: th.Tensor = th.cat(
        [
            classifier(
                th.cat(
                    [
                        results["X"][_ridxs] * results["mask"][_ridxs]
                        - (1 - results["mask"][_ridxs]) * 10,
                        results["mask"][_ridxs],
                    ],
                    dim=1,
                ),
                None,
            )
            for _ridxs in th.split(ridxs, bsz)
        ],
        dim=0,
    )
    # compute metrics
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    metrics_func.reset()
    metrics_func.update(pyhats, th.argmax(results["y"][ridxs], dim=1))
    metrics_d = {k: v.item() for k, v in metrics_func.compute().items()}
    metrics_func.reset()
    # compute feature observed
    metrics_d["feature observed"] = int(
        th.mean(
            th.sum(
                results["mask"][
                    th.argwhere(results["Action"][:, -1] == 1).flatten(), :
                ],
                dim=1,
            )
        ).item()
    )
    # configure loggers
    tfb_logger = plf_loggers.TensorBoardLogger(output_dir, name="", version="")
    csv_logger = plf_loggers.CSVLogger(tfb_logger.log_dir, name="", version="")
    plf = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="cpu")
    # log result
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_val"))
    # flush loggers
    tfb_logger.finalize("success")
    csv_logger.finalize("success")


if __name__ == "__main__":

    @hd.main(version_base=None)
    def _main(cfg: MainConf):
        logger = logging.getLogger(HydraConfig.get().job.name)
        try:
            main(cfg)
        except Exception as e:
            logger.error(e, exc_info=True, stack_info=True)
            traceback.print_exception(e)

    _main()
