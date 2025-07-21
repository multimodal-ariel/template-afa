"""Torch specific metrics that act on tensors.

NOTE: Classification metrics assume that y_pred is a probability distribution
over classes.
They use the library torchmetrics which splits into multiclass and binary
metrics. These metrics have conditions that account for binary or multiclass.
Classification metrics are accuracy, auroc, auprc, f1_score.
"""

import torch
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_average_precision,
    binary_f1_score,
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_average_precision,
    multiclass_f1_score,
)


def torch_accuracy(y_pred, y):
    if y_pred.ndim == 2:
        if y_pred.shape[1] > 2:
            return multiclass_accuracy(
                y_pred, y, num_classes=y_pred.shape[1], average="micro"
            ).item()
        y_pred = y_pred[:, -1]
    return binary_accuracy(y_pred, y).item()


def torch_auroc(y_pred, y):
    if y_pred.ndim == 2:
        if y_pred.shape[1] > 2:
            return multiclass_auroc(
                y_pred, y, num_classes=y_pred.shape[1], average="macro"
            ).item()
        y_pred = y_pred[:, -1]
    return binary_auroc(y_pred, y).item()


def torch_auprc(y_pred, y):
    if y_pred.ndim == 2:
        if y_pred.shape[1] > 2:
            auprc_scores = multiclass_average_precision(
                y_pred, y, num_classes=y_pred.shape[1], average=None
            )
            return torch.mean(torch.nan_to_num(auprc_scores, nan=0.0)).item()
        y_pred = y_pred[:, -1]
    return torch.nan_to_num(binary_average_precision(y_pred, y), nan=0.0).item()


def torch_f1_score(y_pred, y):
    if y_pred.ndim == 2:
        if y_pred.shape[1] > 2:
            return multiclass_f1_score(
                y_pred, y, num_classes=y_pred.shape[1], average="micro"
            ).item()
        y_pred = y_pred[:, -1]
    return binary_f1_score(y_pred, y).item()


# Dictionary where we can choose the metrics by name.
metrics_dict = {
    "accuracy": torch_accuracy,
    "auroc": torch_auroc,
    "auprc": torch_auprc,
    "f1_score": torch_f1_score,
}
