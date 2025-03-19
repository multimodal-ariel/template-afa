import torch
import torch.nn as nn

from .utils import generate_random_feature_mask, mask_batch


def assign_metric(model):
    model.metric = "accuracy" if model.data_parameters["n_classes"] > 1 else "-mse"
    if model.params.use_f1_score:
        model.metric = "f1_score"


def finetune_step(model, batch, is_eval):
    batch = batch["batch"]
    if is_eval:
        unobserved_mask = generate_random_feature_mask(batch, mask_type="full")
    else:
        unobserved_mask = generate_random_feature_mask(batch)
    with torch.no_grad():
        if model.params.problem == "difa":
            futures = model.compute_expected_future_state(batch, unobserved_mask)
            state = model.compute_state(batch, unobserved_mask, futures)
        elif model.params.problem in {"gsmrl", "jafa"}:
            state, _, _ = model.compute_state(batch, unobserved_mask)
        elif model.params.problem in {"eddi", "random"}:
            state = model.compute_state(batch, unobserved_mask)
    if is_eval:
        with torch.no_grad():
            output = model.pred_model(state)
    else:
        output = model.pred_model(state)
    loss = compute_loss(model, output, batch)
    if not is_eval:
        model.optimizer.zero_grad()
        loss.backward()
        if model.params.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.pred_model.parameters(),
                max_norm=model.params.grad_norm,
                norm_type="inf",
            )

        model.optimizer.step()
        model.scheduler.step()
    metric = compute_metric(model, output, batch)
    return {model.metric: metric, "size": len(batch)}


class ConstantLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup = max(int(warmup_epochs), 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [lr / (self.warmup + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warmup:
            c = (self.last_epoch + 1.0) / self.last_epoch
            return [group["lr"] * c for group in self.optimizer.param_groups]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]


def assign_optimizer_scheduler(model, iters=None):
    if iters is None:
        iters = model.params.pretrain_iters
    # use cyclic LR
    model.optimizer = torch.optim.Adam(
        model.pred_model.parameters(),
        lr=model.params.lr,
        weight_decay=model.params.weight_decay,
    )
    model.params.trainset_size = len(model.trainset)

    step_size = (model.params.trainset_size * iters) / (
        model.params.batch_size * model.params.cycle
    )
    model.scheduler = torch.optim.lr_scheduler.CyclicLR(
        model.optimizer,
        base_lr=model.params.base_lr,
        max_lr=model.params.lr,
        cycle_momentum=False,
        mode="triangular2",
        step_size_up=step_size,
    )


def compute_batch_loss(model, output, batch):
    if model.metric in {"accuracy", "f1_score"}:
        cc_batch_loss = nn.CrossEntropyLoss(
            weight=model.params.weight, reduction="none"
        )
        loss = cc_batch_loss(output, batch[:, -1].long())
    else:
        loss = (output.squeeze(1) - batch[:, -1]) ** 2.0
    return loss


def compute_loss(model, output, batch):
    if model.metric in {"accuracy", "f1_score"}:
        cc_loss = nn.CrossEntropyLoss(weight=model.params.weight)
        loss = cc_loss(output, batch[:, -1].long())
    else:
        loss = torch.mean((output.squeeze(1) - batch[:, -1]) ** 2.0)
    return loss


def compute_metric(model, output, batch):
    if model.metric == "accuracy":
        metric = float(
            (torch.argmax(output, dim=-1) == batch[:, -1].long())
            .float()
            .mean()
            .detach()
            .cpu()
        )
    elif model.metric == "-mse":  # regression
        metric = -float(((output - batch[:, -1:]) ** 2.0).mean().detach().cpu())
    elif model.metric == "f1_score":  # "f1_score"
        pred = torch.argmax(output, dim=-1).detach().cpu().numpy()
        target = batch[:, -1].long().detach().cpu().numpy()
        metric = {"y_true": target, "y_pred": pred}
    return metric
