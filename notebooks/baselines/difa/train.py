# %%
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import difalib
import lightning as pl
import mylib
import torch as th
import tqdm.auto as tqdm
from omegaconf import OmegaConf


# %%
@dataclass
class MainConf:
    difa_cfg: DIFAMainConf
    imputation_model_cfg: Optional[ImputationModelConf]
    plf: Any


@dataclass
class DIFAMainConf:
    name: str
    project: str
    cuda: str
    save: bool
    neptune: bool
    nodes: str
    slurm_partition: str
    data: str
    weight: th.Tensor | None
    imputation_model: str
    problem: Literal["vaeac", "eddi", "gsmrl", "jafa", "difa", "random"]
    seed: int
    hidden_dim: int
    embed_dim: int
    depth: int
    iters: int
    resnet: int
    pretrain_iters: int
    workers: int
    batch_size: int
    ppo_batch_size: int
    filters: int
    dropout: float
    min_sigma: float
    gamma: float
    acquisition_cost: float
    n_features: int
    warmup_iters: int
    lr: float
    base_lr: float
    finetune_lr: float
    policy_lr: float
    policy_base_lr: float
    weight_decay: float
    grad_norm: float
    ent_reg: float
    latent_dim: int
    cycle: int
    nosharing: bool
    cost_based: bool
    cnn: bool
    augmentation: bool
    debug: bool
    use_f1_score: bool
    # post init
    pretrain: bool
    use_surrogate_reward: bool
    use_aux_state: bool
    use_imputation_model: bool
    output_dir: str


@dataclass
class ImputationModelConf:
    exp_p: str
    run_id: int
    model_p: str


def make_default_difa_cfg() -> DIFAMainConf:
    args = DIFAMainConf(
        name="demo2",
        project="difa",
        cuda="cpu",
        save=True,
        neptune=False,
        nodes="",
        slurm_partition="",
        data="physio-mortality",
        weight=None,
        imputation_model="models/demo2.pt",
        problem="jafa",
        seed=279,
        hidden_dim=256,
        embed_dim=32,
        depth=2,
        iters=1000,
        resnet=18,
        pretrain_iters=99,
        workers=0,
        batch_size=512,
        ppo_batch_size=512,
        filters=32,
        dropout=0.05,
        min_sigma=0.1,
        gamma=0.999,
        acquisition_cost=0.05,
        n_features=7,
        warmup_iters=20,
        lr=2e-4,
        base_lr=1e-6,
        finetune_lr=1e-6,
        policy_lr=2e-4,
        policy_base_lr=2e-4,
        weight_decay=1e-8,
        grad_norm=10.0,
        ent_reg=1e-3,
        latent_dim=64,
        cycle=10,
        nosharing=True,
        cost_based=False,
        cnn=False,
        augmentation=False,
        debug=False,
        use_f1_score=False,
        pretrain=False,
        use_surrogate_reward=False,
        use_aux_state=True,
        use_imputation_model=True,
        output_dir="outputs",
    )
    return args


def override_difa_cfg_(
    difa_args: DIFAMainConf, jafa_cfg: Any, output_dir: str, plf: pl.Fabric
) -> DIFAMainConf:
    # copy from hydra config to jafa args
    for _k in OmegaConf.to_container(jafa_cfg).keys():
        assert hasattr(difa_args, _k), f"{_k} is an invalid jafa argument"
    for k in asdict(difa_args):
        if hasattr(jafa_cfg, k):
            setattr(difa_args, k, getattr(jafa_cfg, k))
    if plf.device.type == "cuda":
        difa_args.cuda = plf.device.type
    difa_args.output_dir = os.path.join(output_dir, "outputs")
    os.makedirs(difa_args.output_dir, exist_ok=True)
    difa_args.pretrain = False if difa_args.problem in {"vaeac"} else True
    if difa_args.weight is not None:
        # make 1/class_freq to 1/2*class_freq
        # difa_args.weight = (
        #     th.tensor([float(d) for d in difa_args.weight.split()]).to(plf.device) / 2.0
        # )
        difa_args.weight = th.tensor(difa_args.weight, device=plf.device) / 2.0
    else:
        difa_args.weight = None
    difa_args.use_aux_state = False
    # Add Model Specific params
    if difa_args.problem == "gsmrl":
        difa_args.use_surrogate_reward = True
        difa_args.use_aux_state = True
        difa_args.use_imputation_model = True
    if difa_args.problem == "jafa":
        difa_args.use_surrogate_reward = False
        difa_args.use_aux_state = False
        difa_args.use_imputation_model = False
    if difa_args.problem == "difa":
        difa_args.use_imputation_model = True
        difa_args.use_aux_state = True
    if difa_args.problem == "random":
        difa_args.use_imputation_model = True
        difa_args.use_aux_state = False
    return difa_args


def configure_difa_cfg_imputation_model_(difa_args: DIFAMainConf, cfg: MainConf):
    if (not hasattr(cfg, "imputation_model_cfg")) or cfg.imputation_model_cfg is None:
        return difa_args
    assert hasattr(cfg, "imputation_model_cfg") and cfg.imputation_model_cfg is not None
    difa_args.imputation_model = os.path.join(
        mylib.utils.get_project_root_dir(),
        cfg.imputation_model_cfg.exp_p,
        str(cfg.imputation_model_cfg.run_id),
        cfg.imputation_model_cfg.model_p,
    )


def pretrain(model, args, run, plf: pl.Fabric):
    best_logs, best_valid_logs, debug = {}, {}, model.params.debug
    best_pred_model = deepcopy(model.pred_model)
    pbar = tqdm.trange(
        model.params.pretrain_iters, desc="pretrain", dynamic_ncols=True, leave=True
    )
    for cur_iter in pbar:
        train_loader, valid_loader, test_loader = plf.setup_dataloaders(
            *model.dataloaders(iters=cur_iter)
        )
        train_logs = []
        for _, batch in enumerate(train_loader):
            train_logs.append(
                model.finetune_step(
                    difalib.utils.to_device(batch, device=plf.device), is_eval=False
                )
            )
            if debug:
                break
        train_logs = difalib.utils.agg_all_metrics(train_logs, epoch=cur_iter)
        primary_metric = model.metric
        # Validation and Testing
        test_logs, valid_logs = [], []
        for batch in test_loader:
            test_logs.append(
                model.finetune_step(
                    difalib.utils.to_device(batch, device=plf.device), is_eval=True
                )
            )
            if debug:
                break
        for batch in valid_loader:
            valid_logs.append(
                model.finetune_step(
                    difalib.utils.to_device(batch, device=plf.device), is_eval=True
                )
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
        pbar.set_postfix(train_logs)
        plf.log_dict(
            {
                **mylib.utils.add_prefix_to_dict(best_logs, "pbest"),
                **mylib.utils.add_prefix_to_dict(train_logs, "ptrain"),
                **mylib.utils.add_prefix_to_dict(valid_logs, "pval"),
                **mylib.utils.add_prefix_to_dict(test_logs, "ptest"),
            },
            step=cur_iter,
        )
        if args.neptune:
            difalib.utils.log_to_neptune(train_logs, "ptrain", run)
            difalib.utils.log_to_neptune(test_logs, "ptest", run)
            difalib.utils.log_to_neptune(valid_logs, "pvalid", run)
            difalib.utils.log_to_neptune(best_logs, "pbest", run)
        if debug:
            break
    pbar.close()
    model.pred_model.load_state_dict(best_pred_model.state_dict())
    model.optimizer = th.optim.Adam(
        model.pred_model.parameters(),
        lr=model.params.finetune_lr,
        weight_decay=model.params.weight_decay,
    )

    step_size = (model.params.trainset_size * model.params.iters) / (
        model.params.batch_size * model.params.cycle
    )
    model.scheduler = th.optim.lr_scheduler.CyclicLR(
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


def difa_main(difa_cfg: DIFAMainConf | argparse.Namespace, plf: pl.Fabric):
    # args = difalib.utils.parse_params(argv)  # parse arguments
    # device = difalib.utils.get_device(args)
    device = plf.device
    run = None
    # if args.neptune:
    #     run = difalib.utils.set_neptune_experiment(args)
    difalib.utils.set_random_seed(difa_cfg)  # set seed for reproducibility
    difalib.utils.set_logging_directory(difa_cfg)  # set output log path
    # Setup Model and Dataset
    model = difalib.models.REGISTERED_MODELS[difa_cfg.problem](difa_cfg, device=device)
    if difa_cfg.pretrain:
        pretrain(model, difa_cfg, run, plf=plf)
    best_logs, best_valid_logs, debug = {}, {}, model.params.debug
    #
    pbar = tqdm.trange(difa_cfg.iters, desc="main", dynamic_ncols=True, leave=True)
    for cur_iter in pbar:
        train_loader, valid_loader, test_loader = plf.setup_dataloaders(
            *model.dataloaders(iters=cur_iter)
        )
        train_logs = []
        for _, batch in enumerate(train_loader):
            train_logs.append(model.train_step(batch))
            if debug:
                break
        train_logs = difalib.utils.agg_all_metrics(train_logs, epoch=cur_iter)
        primary_metric = difalib.utils.get_primary_metric(difa_cfg.problem)
        if isinstance(primary_metric, list):
            primary_metric = [d for d in primary_metric if d in train_logs][0]
        # Validation and Testing
        test_logs, valid_logs = [], []
        for batch in test_loader:
            test_logs.append(model.test_step(batch))
            if debug:
                break
        for batch in valid_loader:
            valid_logs.append(model.test_step(batch))
            if debug:
                break
        test_logs_ = difalib.utils.agg_all_metrics(test_logs, epoch=cur_iter)
        valid_logs_ = difalib.utils.agg_all_metrics(valid_logs, epoch=cur_iter)
        is_improved = difalib.utils.update_best_logs(
            best_logs, best_valid_logs, valid_logs_, test_logs_, primary_metric
        )
        pbar.set_postfix(train_logs)
        plf.log_dict(
            {
                **mylib.utils.add_prefix_to_dict(best_logs, "best"),
                **mylib.utils.add_prefix_to_dict(train_logs, "train"),
                **mylib.utils.add_prefix_to_dict(valid_logs_, "val"),
                **mylib.utils.add_prefix_to_dict(test_logs_, "test"),
                "is_improved": is_improved,
            },
            step=cur_iter,
        )
        if difa_cfg.neptune:
            difalib.utils.log_to_neptune(train_logs, "train", run)
            difalib.utils.log_to_neptune(test_logs_, "test", run)
            difalib.utils.log_to_neptune(valid_logs_, "valid", run)
            difalib.utils.log_to_neptune(best_logs, "best", run)
        if difa_cfg.save and is_improved:
            model.save()
            th.save(
                {"test_logs": test_logs, "valid_logs": valid_logs},
                os.path.join(difa_cfg.output_dir, f"itr_{cur_iter}.pt"),
            )
        if debug:
            break
    pbar.close()


# %%
# argv = str.split('--data grid --problem difa --lr 0.0001 --policy_lr 0.0002 --policy_base_lr 1e-05 --seed 998 --imputation_model models/2987534.pt --iters 5000 --pretrain_iters 2000 --workers 2 --batch_size 512 --n_features 4 --grad_norm 10.0 --weight none'.split(
# argv = str.split(
#     "--data grid --problem difa --lr 0.0001 --policy_lr 0.0002 --policy_base_lr 1e-05 --seed 998 --iters 5000 --pretrain_iters 2000 --workers 2 --batch_size 512 --n_features 4 --grad_norm 10.0 --weight none"
#     " "
# )

# %%
cfg: MainConf = OmegaConf.load("pretrain.yaml")  # type:ignore
output_dir: str = "outputs"

# %%
plf = pl.Fabric(accelerator="cpu")
difa_args: DIFAMainConf = make_default_difa_cfg()
override_difa_cfg_(
    difa_args=difa_args, jafa_cfg=cfg.difa_cfg, output_dir=output_dir, plf=plf
)

# %%
difa_main(argparse.Namespace(**asdict(difa_args)), plf=plf)

# %%
