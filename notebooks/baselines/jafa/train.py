# %%
from __future__ import annotations

import csv
import os
import pickle as pkl
import pprint
import random
from dataclasses import asdict, dataclass
from typing import Any, Optional

import hydra as hd
import jafalib
import lightning as pl
import numpy as np
import tensordict as thd
import torch as th
from jafalib.model import MLP, DFSNet, DuelingNet, SetEncoder
from omegaconf import OmegaConf

pp = pprint.PrettyPrinter()


# %%
@dataclass
class MainConf:
    data: Any
    jafa_cfg: JAFAMainConf
    plf: pl.Fabric


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
    jafa_args.save_dir = os.path.join(output_dir, "jafa")
    jafa_args.save_path = os.path.join(jafa_args.save_dir, "output")
    os.makedirs(jafa_args.save_path, exist_ok=True)
    jafa_args.csv_path = os.path.join(jafa_args.save_dir, "jafa_metrics")
    # args.save_dir = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), args.save_dir
    # )
    # args.save_path = (
    #     args.data_type
    #     + "_nenv{}_nsteps{}_cost{}_norm{}".format(
    #         args.n_envs, args.nsteps, args.r_cost, args.normalize
    #     )
    #     + "eps_start{}end{}decay{}_".format(
    #         args.eps_start, args.eps_end, args.decay_rate
    #     )
    #     + "complete{}_doneactiontrain{}".format(args.complete, args.done_action_train)
    #     + "emb"
    #     + "_".join(
    #         "%03d" % num for num in args.embedder_hidden_sizes + [args.embedded_dim]
    #     )
    #     + "clf"
    #     + "_".join("%03d" % num for num in args.clf_hidden_sizes)
    #     + "policy"
    #     + "_".join("%03d" % num for num in args.policy_hidden_sizes + [args.shared_dim])
    # )
    # args.save_path = args.save_path + "_batch_size{}".format(args.batch_size)
    # if args.dropout:
    #     args.save_path = args.save_path + "_dropout{}".format(args.p)
    # if len(args.message) > 0:
    #     args.save_path += args.message
    # args.save_path += "lstm{}_".format(args.lstm_size) + "shuffle{}_".format(
    #     args.n_shuffle
    # )
    # if args.pretrain:
    #     args.save_path += "_pretrain{}_{}".format(args.pretrain, args.pretrain_sample)
    # if args.batchnorm:
    #     args.save_path += "_batchnorm"
    # args.save_path = os.path.join(args.save_dir, args.save_path)
    # args.csv_path = args.save_path
    # args.save_path = args.save_path + "seed{}".format(args.random_seed)
    # os.makedirs(args.save_path, exist_ok=True)


def to_jafa_datatmp(data: thd.TensorDict, shuffle: bool) -> jafalib.data_temp.DataTemp:
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


# %%
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
    tstresults_l = list()
    for prefix, environ in [("val", valenv), ("tr", env), ("ts", testenv)]:
        step_runner.load(os.path.join(args.save_path, "trained_best.model"))
        environ.reset()
        # trainset
        if environ is None:
            break
        environ.reset()
        test_result = jafalib.main.test(step_runner, environ, args)
        tstresults_l.append(test_result)
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
    file_exists = os.path.isfile(args.csv_path + ".csv")
    with open(args.csv_path + ".csv", "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    with open(args.csv_path + ".pkl", mode="wb") as f:
        pkl.dump(tstresults_l, f)


# %%
def jafa_main(
    args,
    traindata: jafalib.data_temp.DataTemp,
    valdata: jafalib.data_temp.DataTemp,
    testdata: jafalib.data_temp.DataTemp,
):
    embedder_hidden_sizes = args.embedder_hidden_sizes
    embedded_dim = args.embedded_dim
    lstm_size = args.lstm_size
    n_shuffle = args.n_shuffle
    clf_hidden_sizes = args.clf_hidden_sizes
    policy_hidden_sizes = args.policy_hidden_sizes
    shared_dim = args.shared_dim
    nsteps = args.nsteps
    n_envs = args.n_envs
    r_cost = args.r_cost

    # TODO data load first, classifier defining and declare env
    # (traindata, valdata, testdata), cost = jafalib.data_temp.data_load(
    #     data_type=args.data_type,
    #     random_seed=args.random_seed,
    #     cost_from_file=args.cost_from_file,
    # )
    cost = None
    if cost is not None:
        r_cost = cost
    input_dim = traindata.n_features + 1
    clf_output_size = traindata.n_classes if traindata.n_classes > 2 else 1
    encoder = SetEncoder(
        input_dim,
        traindata.n_features,
        embedder_hidden_sizes,
        embedded_dim,
        lstm_size,
        n_shuffle,
        normalize=args.normalize,
        dropout=args.dropout,
        p=args.p,
    )

    dfsnet = DFSNet(
        encoder=encoder,
        classifier=MLP(
            lstm_size + embedded_dim,
            clf_hidden_sizes,
            clf_output_size,
            dropout=args.dropout,
            p=args.p,
            batch_norm=args.batchnorm,
        ),
        policy=DuelingNet(
            lstm_size + embedded_dim,
            policy_hidden_sizes,
            shared_dim,
            traindata.n_actions,
        ),
    )
    dfsnet.to(args.device)
    step_runner = jafalib.main.StepRunner(dfsnet, args)
    env = jafalib.environment.Env(
        args,
        n_envs,
        r_cost,
        traindata,
        step_runner.classify,
    )
    valenv = jafalib.environment.Env(
        args,
        n_envs,
        r_cost,
        valdata,
        step_runner.classify,
    )
    testenv = jafalib.environment.Env(
        args,
        n_envs,
        r_cost,
        testdata,
        step_runner.classify,
    )

    env.classify = step_runner.classify
    valenv.classify = step_runner.classify
    testenv.classify = step_runner.classify
    jafalib.main.learn(
        step_runner,
        args,
        env,
        valenv,
        nsteps=nsteps,
        total_steps=int(5e6),
        scheduler=args.scheduler,
    )
    dfsnet.eval()
    test_and_record(step_runner, args, env, valenv, testenv)


# %%
cfg: MainConf = OmegaConf.load("test.yaml")  # type:ignore

# %%
plf: pl.Fabric = pl.Fabric()

# %%
jafa_args = make_default_jafa_cfg()
override_jafa_cfg_(
    jafa_args=jafa_args,
    jafa_cfg=cfg.jafa_cfg,
    output_dir=os.path.join("outputs", f"{cfg.data.name}"),
    plf=plf,
)

# %%
random.seed(jafa_args.random_seed)
np.random.seed(jafa_args.random_seed)
th.manual_seed(jafa_args.random_seed)

# %%
# TODO make dataset function
tdata, vdata, tstdata = hd.utils.instantiate(cfg.data)
tjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(tdata, True)
vjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(vdata, False)
tstjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(tstdata, False)

# %%
jafa_main(jafa_args, tjdatatmp, vjdatatmp, tstjdatatmp)

# %%
# tdata =

# %%
# parser = argparse.ArgumentParser()
# parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
# parser.add_argument(
#     "--complete",
#     action="store_true",
#     help="train \
#         classifier with complete data",
# )
# parser.add_argument(
#     "--pretrain",
#     help="pre classifier training",
#     type=int,
#     default=10000,
# )
# parser.add_argument("--pretrain_sample", help="", type=str, default="both")
# parser.add_argument("--mode", help="double dqn?", type=str, default="double")
# parser.add_argument("--scheduler", help="ent_coef", type=str, default="linear")
# parser.add_argument("--dropout", action="store_true", help="Dropout classifier")
# parser.add_argument("--batchnorm", action="store_true", help="batch norm")
# parser.add_argument(
#     "--done_action_train", action="store_true", help="done action train"
# )
# parser.add_argument("--data_type", help="data", type=str, default="cube_20_0.3")
# parser.add_argument("--p", help="dropout prob", type=float, default=0)
# parser.add_argument(
#     "--group_norm", type=float, default=0, help="group_norm regularization param"
# )
# parser.add_argument(
#     "--save_dir", help="save directory name", type=str, default="result"
# )
# parser.add_argument(
#     "--embedder_hidden_sizes", help="embedder", type=str, default="[32, 32]"
# )
# parser.add_argument(
#     "--clf_hidden_sizes", help="clf mlp size", type=str, default="[32, 32]"
# )
# parser.add_argument(
#     "--policy_hidden_sizes", help="a2c mlp size", type=str, default="[32]"
# )
# parser.add_argument(
#     "--shared_dim",
#     help="a2c net shared vertor dim for pi and v",
#     type=int,
#     default="16",
# )
# parser.add_argument("--target_update_freq", help=".", type=int, default=100)
# parser.add_argument("--eps_start", help=".", type=float, default=1.0)
# parser.add_argument("--eps_end", help=".", type=float, default=0.1)
# parser.add_argument("--decay_rate", help=".", type=float, default=2)
# parser.add_argument(
#     "--n_envs", help="how many episodes simultaneouly?", type=int, default=128
# )
# parser.add_argument(
#     "--nsteps", help="num of steps for calc return", type=int, default=4
# )
# parser.add_argument(
#     "--normalize",
#     help="make embedded feature l2 norm to 1",
#     type=bool,
#     default=True,
# )
# parser.add_argument(
#     "--embedded_dim", help="embedded vector dimension", type=int, default=16
# )
# parser.add_argument("--lstm_size", help="encoder lstm size", type=int, default=16)
# parser.add_argument("--n_shuffle", help="n shuffle", type=int, default=5)
# parser.add_argument(
#     "--r_cost", help="cost weight(negative value)", default=-0.05, type=float
# )
# parser.add_argument(
#     "--cost_from_file",
#     help="whether the cost info is in data csv file or not",
#     type=bool,
#     default=False,
# )
# parser.add_argument("--random_seed", help="random seed", type=int, default=123)
# parser.add_argument("--batch_size", help="batch size", type=int, default=128)
# parser.add_argument("--message", help="message", type=str, default="")
# args = parser.parse_args(["--disable_cuda"])

# args.clf_hidden_sizes = eval(args.clf_hidden_sizes)
# args.policy_hidden_sizes = eval(args.policy_hidden_sizes)
# args.embedder_hidden_sizes = eval(args.embedder_hidden_sizes)

# random.seed(args.random_seed)
# np.random.seed(args.random_seed)
# th.manual_seed(args.random_seed)

# if not args.disable_cuda and th.cuda.is_available():
#     args.device = th.device("cuda")
#     th.cuda.manual_seed(args.random_seed)
# else:
#     args.device = th.device("cpu")
# args.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_dir)

# args.save_path = (
#     args.data_type
#     + "_nenv{}_nsteps{}_cost{}_norm{}".format(
#         args.n_envs, args.nsteps, args.r_cost, args.normalize
#     )
#     + "eps_start{}end{}decay{}_".format(args.eps_start, args.eps_end, args.decay_rate)
#     + "complete{}_doneactiontrain{}".format(args.complete, args.done_action_train)
#     + "emb"
#     + "_".join("%03d" % num for num in args.embedder_hidden_sizes + [args.embedded_dim])
#     + "clf"
#     + "_".join("%03d" % num for num in args.clf_hidden_sizes)
#     + "policy"
#     + "_".join("%03d" % num for num in args.policy_hidden_sizes + [args.shared_dim])
# )
# args.save_path = args.save_path + "_batch_size{}".format(args.batch_size)
# if args.dropout:
#     args.save_path = args.save_path + "_dropout{}".format(args.p)
# if len(args.message) > 0:
#     args.save_path += args.message
# args.save_path += "lstm{}_".format(args.lstm_size) + "shuffle{}_".format(args.n_shuffle)
# if args.pretrain:
#     args.save_path += "_pretrain{}_{}".format(args.pretrain, args.pretrain_sample)
# if args.batchnorm:
#     args.save_path += "_batchnorm"
# args.save_path = os.path.join(args.save_dir, args.save_path)
# args.csv_path = args.save_path
# args.save_path = args.save_path + "seed{}".format(args.random_seed)
# os.makedirs(args.save_path, exist_ok=True)
# jafa_officail_main_fn(args)

# # %%
