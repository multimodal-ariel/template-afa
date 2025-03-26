import argparse
import json
import os
import random
import subprocess
from typing import Optional, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.metrics import f1_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def log_to_neptune(logs, split, run, final=False):
    for k, v in logs.items():
        if final:
            run["{}/{}".format(split, k)] = v
        else:
            run["{}/{}".format(split, k)].log(v)


def get_primary_metric(problem):
    with open("references/metrics.json", "r") as fp:
        metrics = json.load(fp)
    return metrics.get(problem, None)


def update_best_logs(best_logs, best_valid_logs, valid_logs, test_logs, primary_metric):
    is_improved = False
    if primary_metric in valid_logs and valid_logs[
        primary_metric
    ] > best_valid_logs.get(primary_metric, float("-inf")):
        for k in test_logs.keys():
            best_logs[k] = test_logs[k]
            best_valid_logs[k] = valid_logs[k]
        is_improved = True
    return is_improved


def agg_all_metrics(logs, epoch=None):
    output = {}
    if len(logs) == 0:
        return output
    # either value or a list
    keys = [
        k for k in logs[0].keys() if not isinstance(logs[0][k], dict) and k != "size"
    ]

    is_size_available = "size" in logs[0]
    for k in keys:
        # B, or B,D
        all_logs = torch.tensor([d[k] for d in logs])
        counts = torch.tensor(
            [d["size"] if is_size_available else 1 for d in logs]
        ).float()[:, None]
        if len(all_logs.shape) == 1:
            all_logs.unsqueeze_(1)
        all_logs = (
            (torch.sum(all_logs * counts, dim=0) / torch.sum(counts, dim=0))
            .numpy()
            .tolist()
        )
        if len(all_logs) > 1 and len(all_logs) < 101:
            output[k + "_list"] = all_logs
        output[k] = all_logs[-1]
    # Custom dictionaries
    keys = [k for k in logs[0].keys() if "f1_score" in k]
    for k in keys:
        try:
            y_pred = np.concatenate([d[k]["y_pred"] for d in logs])
            y_true = np.concatenate([d[k]["y_true"] for d in logs])
            output[k] = f1_score(y_true, y_pred)
        except TypeError:
            output[k] = -1.0
    if epoch is not None:
        output["epoch"] = epoch
    return output


def get_device(args):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and args.cuda
        else torch.device("cpu")
    )
    if args.cuda:
        if device.type != "cuda":
            time_now = (
                subprocess.check_output("date", shell=True).decode("utf-8").strip("\n")
            )
            command = "sacct -n -j " + args.name + " --format=Jobname%100"
            config_name = [
                d
                for d in (subprocess.check_output(command, shell=True))
                .decode("utf-8")
                .split(" ")
                if ".sh" in d
            ][0]
            line = ",".join([args.name, args.nodes, time_now, config_name, "\n"])
            with open("exceptions.txt", "a") as fp:
                fp.write(line)
            raise ValueError("No GPU Found!")
    return device


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="demo2", help="name of the experiment")
    parser.add_argument("--project", default="arighosh/difa", help="neptune project")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--neptune", action="store_true")
    parser.add_argument("--nodes", default="", help="slurm nodes for the experiment")
    parser.add_argument(
        "--slurm_partition", default="", help="slurm partitions for the experiment"
    )
    parser.add_argument("--data", default="physio-mortality", help="Dataset")
    parser.add_argument("--weight", default="none", help="cc loss weight")
    parser.add_argument(
        "--imputation_model", default="models/demo2.pt", help="VAEAC Imputation Model"
    )
    parser.add_argument("--problem", default="jafa", help="The problem to train")
    parser.add_argument("--seed", default=999, type=int, help="Random seed")
    parser.add_argument("--hidden_dim", default=256, type=int, help="hidden dim")
    parser.add_argument("--embed_dim", default=32, type=int, help="embedding dim")
    parser.add_argument(
        "--depth",
        default=2,
        type=int,
        help="depth of skip\
            connection",
    )

    parser.add_argument("--iters", default=1000, type=int, help="Number of Iteration")
    parser.add_argument(
        "--resnet",
        default=18,
        type=int,
        help="Resnet18 or\
            Resnet9",
    )
    parser.add_argument(
        "--pretrain_iters", default=99, type=int, help="Number of Pretrain Iteration"
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="The number of data loader workers"
    )
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--ppo_batch_size", default=512, type=int)
    parser.add_argument("--filters", default=32, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--min_sigma", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.999, type=float)
    parser.add_argument("--acquisition_cost", default=0.05, type=float)
    parser.add_argument("--n_features", default=7, type=int)
    parser.add_argument("--warmup_iters", default=20, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--base_lr", default=1e-6, type=float)
    parser.add_argument("--finetune_lr", default=1e-6, type=float)
    parser.add_argument("--policy_lr", default=2e-4, type=float)
    parser.add_argument("--policy_base_lr", default=2e-4, type=float)

    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument("--grad_norm", default=10.0, type=float)
    parser.add_argument("--ent_reg", default=1e-3, type=float)

    parser.add_argument("--latent_dim", default=64, type=int)
    parser.add_argument("--cycle", default=10, type=int)
    parser.add_argument("--nosharing", action="store_false")
    parser.add_argument("--cost_based", action="store_true")
    parser.add_argument("--cnn", action="store_true")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_f1_score", action="store_true")
    return parser


def parse_params(
    args: Optional[Sequence[str]] = None,
):
    parser: argparse.ArgumentParser = get_parser()
    params = parser.parse_args(args)
    params.pretrain = False if params.problem in {"vaeac"} else True
    if params.weight != "none":
        # make 1/class_freq to 1/2*class_freq
        params.weight = (
            torch.tensor([float(d) for d in params.weight.split()]).to(device) / 2.0
        )
    else:
        params.weight = None

    params.use_aux_state = False
    # Add Model Specific params
    if params.problem == "gsmrl":
        params.use_surrogate_reward = True
        params.use_aux_state = True
        params.use_imputation_model = True
    if params.problem == "jafa":
        params.use_surrogate_reward = False
        params.use_aux_state = False
        params.use_imputation_model = False

    if params.problem == "difa":
        params.use_imputation_model = True
        params.use_aux_state = True
    if params.problem == "random":
        params.use_imputation_model = True
        params.use_aux_state = False
    return params


def set_neptune_experiment(args):
    import neptune.new as neptune

    project = args.project
    run = neptune.init(
        project=project,
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        capture_hardware_metrics=False,
        name=args.name,
    )
    run["parameters"] = vars(args)
    return run


def set_random_seed(args):
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


def set_logging_directory(args):
    args.root = "logs/{}/".format(args.name)
    create_safe_directory(args.root)
    with open(args.root + "config.yml", "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def create_safe_directory(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def to_device(batch):
    batch = {k: x.to(device) if torch.is_tensor(x) else x for k, x in batch.items()}
    return batch
