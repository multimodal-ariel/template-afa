from __future__ import annotations

import csv
import logging
import os
import pickle as pkl
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Optional

import hydra as hd
import jafalib
import lightning as pl
import mylib
import numpy as np
import tensordict as thd
import torch as th
from hydra.core.hydra_config import HydraConfig
from jafalib.model import MLP, DFSNet, DuelingNet, SetEncoder
from omegaconf import OmegaConf


@dataclass
class MainConf:
    train_exp: Optional[MakeTemplateExpConf]
    train_run: Optional[str]
    n_instances: int
    plf: pl.Fabric


@dataclass
class JAFATrainConf:
    data: Any
    jafa_cfg: JAFAMainConf
    plf: pl.Fabric


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


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


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


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


def _jafa_test_runtime(
    run_p: str,
    step_runner: jafalib.main.StepRunner,
    args,
    env: jafalib.environment.Env,
    valenv: jafalib.environment.Env,
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
    prefix = "val"
    environ = valenv
    step_runner.load(os.path.join(args.save_path, "trained_best.model"))
    start_time_ns: int = time.time_ns()
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
    end_time_ns: int = time.time_ns()
    result["inference_time_ns"] = end_time_ns - start_time_ns
    result["avg_pred_time_ns"] = (end_time_ns - start_time_ns) / len(valenv.n_data)
    field = sorted_argskey + field
    result.update(argsdict)
    file_exists = os.path.isfile(args.csv_path + ".csv")
    with open(os.path.join(run_p, "metrics.csv"), "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    with open(os.path.join(run_p, "result.pkl"), mode="wb") as f:
        pkl.dump(tstresults_l, f)


def jafa_log_runtime(
    run_p: str,
    args,
    traindata: jafalib.data_temp.DataTemp,
    valdata: jafalib.data_temp.DataTemp,
):
    embedder_hidden_sizes = args.embedder_hidden_sizes
    embedded_dim = args.embedded_dim
    lstm_size = args.lstm_size
    n_shuffle = args.n_shuffle
    clf_hidden_sizes = args.clf_hidden_sizes
    policy_hidden_sizes = args.policy_hidden_sizes
    shared_dim = args.shared_dim
    nsteps = args.nsteps
    n_envs = 1
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
    valenv.classify = step_runner.classify
    dfsnet.eval()
    _jafa_test_runtime(run_p, step_runner, args, env, valenv)


def _get_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "train_run") and cfg.train_run is not None:
        return cfg.train_run
    assert hasattr(cfg, "train_exp") and cfg.train_exp is not None
    return os.path.join(cfg.train_exp.exp_p, str(cfg.train_exp.run_id))


def main(cfg: MainConf):
    output_dir: str = HydraConfig.get().runtime.output_dir
    run_p: str = _get_run_dir(cfg)
    run_cfg: JAFATrainConf = OmegaConf.load(
        os.path.join(mylib.utils.get_project_root_dir(), run_p, ".hydra", "config.yaml")
    )  # type:ignore
    # instantiate dataset
    tdata, vdata, tstdata = hd.utils.instantiate(run_cfg.data)
    vdata = vdata[: cfg.n_instances]
    tjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(tdata, True)
    vjdatatmp: jafalib.data_temp.DataTemp = to_jafa_datatmp(vdata, False)
    # configure jafa configs
    plf: pl.Fabric = hd.utils.instantiate(run_cfg.plf, _partial_=True)()
    jafa_args: JAFAMainConf = make_default_jafa_cfg()
    override_jafa_cfg_(
        jafa_args=jafa_args, jafa_cfg=run_cfg.jafa_cfg, output_dir=output_dir, plf=plf
    )
    # run main function of jafa
    jafa_log_runtime(run_p, jafa_args, traindata=tjdatatmp, valdata=vjdatatmp)


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
