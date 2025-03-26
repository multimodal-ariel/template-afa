# %%
from __future__ import annotations

import pprint
from copy import deepcopy

import difalib
import torch


# %%
def pretrain(model, args, run):
    best_logs, best_valid_logs, debug = {}, {}, model.params.debug
    best_pred_model = deepcopy(model.pred_model)
    for cur_iter in range(model.params.pretrain_iters):
        train_loader, valid_loader, test_loader = model.dataloaders(iters=cur_iter)
        train_logs = []
        for _, batch in enumerate(train_loader):
            train_logs.append(
                model.finetune_step(difalib.utils.to_device(batch), is_eval=False)
            )
            if debug:
                break
        train_logs = difalib.utils.agg_all_metrics(train_logs, epoch=cur_iter)
        primary_metric = model.metric
        # Validation and Testing
        test_logs, valid_logs = [], []
        for batch in test_loader:
            test_logs.append(
                model.finetune_step(difalib.utils.to_device(batch), is_eval=True)
            )
            if debug:
                break
        for batch in valid_loader:
            valid_logs.append(
                model.finetune_step(difalib.utils.to_device(batch), is_eval=True)
            )
            if debug:
                break
        test_logs = difalib.utils.agg_all_metrics(test_logs, epoch=cur_iter)
        valid_logs = difalib.utils.agg_all_metrics(valid_logs, epoch=cur_iter)
        is_improved = difalib.utils.update_best_logs(
            best_logs, best_valid_logs, valid_logs, test_logs, primary_metric
        )
        if is_improved:
            best_pred_model = deepcopy(model.pred_model)
        if args.neptune:
            difalib.utils.log_to_neptune(train_logs, "ptrain", run)
            difalib.utils.log_to_neptune(test_logs, "ptest", run)
            difalib.utils.log_to_neptune(valid_logs, "pvalid", run)
            difalib.utils.log_to_neptune(best_logs, "pbest", run)
        else:
            pprint.pprint(["pbest_logs", best_logs])
            pprint.pprint(["ptrain_logs", train_logs])
            pprint.pprint(["ptest_logs", test_logs])
        if debug:
            break
    model.pred_model.load_state_dict(best_pred_model.state_dict())
    model.optimizer = torch.optim.Adam(
        model.pred_model.parameters(),
        lr=model.params.finetune_lr,
        weight_decay=model.params.weight_decay,
    )

    step_size = (model.params.trainset_size * model.params.iters) / (
        model.params.batch_size * model.params.cycle
    )
    model.scheduler = torch.optim.lr_scheduler.CyclicLR(
        model.optimizer,
        base_lr=model.params.base_lr,
        max_lr=model.params.finetune_lr,
        cycle_momentum=False,
        mode="triangular",
        step_size_up=step_size,
    )
    if model.params.augmentation:
        model.params.augmentation = False  # Set it before policies
        model.prepare_data()


# %%
# argv = str.split('--data grid --problem difa --lr 0.0001 --policy_lr 0.0002 --policy_base_lr 1e-05 --seed 998 --imputation_model models/2987534.pt --iters 5000 --pretrain_iters 2000 --workers 2 --batch_size 512 --n_features 4 --grad_norm 10.0 --weight none'.split(
argv = str.split(
    "--data grid --problem difa --lr 0.0001 --policy_lr 0.0002 --policy_base_lr 1e-05 --seed 998 --iters 5000 --pretrain_iters 2000 --workers 2 --batch_size 512 --n_features 4 --grad_norm 10.0 --weight none"
    " "
)

# %%
# Setup project
args = difalib.utils.parse_params(argv)  # parse arguments
device = difalib.utils.get_device(args)
run = None
if args.neptune:
    run = difalib.utils.set_neptune_experiment(args)
difalib.utils.set_random_seed(args)  # set seed for reproducibility
difalib.utils.set_logging_directory(args)  # set output log path

# %%
# Setup Model and Dataset
model = difalib.models.REGISTERED_MODELS[args.problem](args, device=device)
if args.pretrain:
    pretrain(model, args, run)
best_logs, best_valid_logs, debug = {}, {}, model.params.debug
#
for cur_iter in range(args.iters):
    train_loader, valid_loader, test_loader = model.dataloaders(iters=cur_iter)
    train_logs = []
    for _, batch in enumerate(train_loader):
        train_logs.append(model.train_step(difalib.utils.to_device(batch)))
        if debug:
            break
    train_logs = difalib.utils.agg_all_metrics(train_logs, epoch=cur_iter)
    primary_metric = difalib.utils.get_primary_metric(args.problem)
    if isinstance(primary_metric, list):
        primary_metric = [d for d in primary_metric if d in train_logs][0]
    # Validation and Testing
    test_logs, valid_logs = [], []
    for batch in test_loader:
        test_logs.append(model.test_step(difalib.utils.to_device(batch)))
        if debug:
            break
    for batch in valid_loader:
        valid_logs.append(model.test_step(difalib.utils.to_device(batch)))
        if debug:
            break
    test_logs = difalib.utils.agg_all_metrics(test_logs, epoch=cur_iter)
    valid_logs = difalib.utils.agg_all_metrics(valid_logs, epoch=cur_iter)
    is_improved = difalib.utils.update_best_logs(
        best_logs, best_valid_logs, valid_logs, test_logs, primary_metric
    )
    if args.neptune:
        difalib.utils.log_to_neptune(train_logs, "train", run)
        difalib.utils.log_to_neptune(test_logs, "test", run)
        difalib.utils.log_to_neptune(valid_logs, "valid", run)
        difalib.utils.log_to_neptune(best_logs, "best", run)
    else:
        pprint.pprint(["best_logs", best_logs])
        pprint.pprint(["train_logs", train_logs])
        pprint.pprint(["test_logs", test_logs])
    if args.save and is_improved:
        model.save()
    if debug:
        break

# %%
