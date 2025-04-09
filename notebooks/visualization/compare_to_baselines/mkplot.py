# %%
from __future__ import annotations

import itertools as itrtls
import logging
import os
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import mylib
import pandas as pd
from omegaconf import OmegaConf

# %%
lgr = logging.getLogger()

# %%
OUTPUT_P = "outputs/proto"
os.makedirs(OUTPUT_P, exist_ok=True)

# %%
# title = "cube"
# exp_ps: dict[str, str] = {
#     "aco": "experiments/baselines/aco/outputs/cube/20250311_201540",
#     "dime": "experiments/baselines/dime/outputs/cube_eval/20250323_173822",
#     "tafa": "experiments/make_template/outputs/cube/20250318_225416",
# }
# xlim: Optional[tuple[float, float]] = (4, 7.5)
# ylim: Optional[tuple[float, float]] = (0.45, 0.9)
# title = "gas"
# exp_ps: dict[str, str] = {
#     "aco": "experiments/baselines/aco/outputs/gas/20250312_143952",
#     # "dime": "experiments/baselines/dime/outputs/gas_eval/20250319_150025",
#     "tafa": "experiments/make_template/outputs/gas_cnnet/20250324_224734",
#     # "tafa": "experiments/make_template/outputs/gas_cnnet/20250314_113723",
#     # "aco": "experiments/baselines/aco/outputs/gas/20250312_143952",
#     # "dime": "experiments/baselines/dime/outputs/gas_eval/20250319_150025",
#     # "tafa": "experiments/make_template/outputs/gas_cnnet/20250314_113723",
# }
# xlim: Optional[tuple[float, float]] = (None, None)
# ylim: Optional[tuple[float, float]] = None
# title = "big5_cnnet"
# exp_ps: dict[str, str] = {
#     "aco": "experiments/baselines/aco/outputs/big5/20250313_153149",
#     "dime": "experiments/baselines/dime/outputs/big5_eval/20250319_144337",
#     # "tafa": "experiments/make_template/outputs/big5/20250305_145410",
#     "tafa": "experiments/make_template/outputs/big5_cnnet/20250318_144121",
# }
# xlim: Optional[tuple[float, float]] = (0.0, 17.0)
# title = "mnist_cnnet"
# exp_ps: dict[str, str] = {
#     "aco": "experiments/baselines/aco/outputs/mnist/20250312_143945",
#     "dime": "experiments/baselines/dime/outputs/mnist_eval/20250323_170930",
#     "tafa": "experiments/make_template/outputs/mnist_cnnet/20250314_113824",
# }
# xlim: Optional[tuple[float, float]] = (3, 50)
# ylim: Optional[tuple[float, float]] = (0.6, 0.95)


# %%
def safe_str_to_int(s: str):
    try:
        return int(s)
    except ValueError as e:
        return -1


def load_baseline_metrics(exp_p: str):
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        metrics_l.append(metrics_df)
    return metrics_l


def load_tafa_metrics(
    exp_p: str, exclude_mktmplfn_name: Optional[list[str]]
) -> dict[str, list[pd.DataFrame]]:
    metrics_d: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
        mktmpl_fn_name: str = str.split(run_cfg.make_templates_fn._target_, ".")[-1]
        if (
            exclude_mktmplfn_name is not None
            and mktmpl_fn_name in exclude_mktmplfn_name
        ):
            continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        if (
            "eval_val/feature observed" not in metrics_df
            and "eval/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue
        metrics_d[mktmpl_fn_name].append(metrics_df)
    return metrics_d


def load_metrics(
    exp_ps: dict[str, str], load_kwargs: Optional[dict[str, Any]]
) -> dict[str, list[pd.DataFrame]]:
    metrics_d: dict[str, list[pd.DataFrame]] = dict()
    for name, exp_p in exp_ps.items():
        if name in {"aco", "dime"}:
            metrics_d[name] = load_baseline_metrics(exp_p)
        elif name == "tafa":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_metrics(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[f"tafa-{_name}"] = _metrics_l
    return metrics_d


def make_plots(metrics_d: dict[str, list[pd.DataFrame]]):
    fig, ax = plt.subplots()
    ax.set_title(title)
    for name, metrics_df_l in metrics_d.items():
        prefix: str
        if "eval/acc" in metrics_df_l[0]:
            prefix = "eval"
        elif "eval_val/acc" in metrics_df_l[0]:
            prefix = "eval_val"
        else:
            logging.warning(f"{name} has unsupported metrics key")
            continue
        data = pd.concat(
            [
                _mdf[[f"{prefix}/feature observed", f"{prefix}/acc"]].dropna()
                for _mdf in metrics_df_l
            ]
        ).to_numpy()
        ax.plot(data[:, 0], data[:, 1], label=name, marker="x")
    ax.set_xlabel("feature observed")
    ax.set_ylabel("accuracy")
    ax.legend()
    return fig, ax


# %%
metrics_d: dict[str, list[pd.DataFrame]] = load_metrics(
    exp_ps=exp_ps,
    load_kwargs={"tafa": {"exclude_mktmplfn_name": ["make_templates_reduce_features"]}},
)

# %%
fig, ax = make_plots(metrics_d=metrics_d)
ax.set_box_aspect(1.0)
if xlim is not None:
    ax.set_xlim(*xlim)
if ylim is not None:
    ax.set_ylim(*ylim)
fig.savefig(f"{OUTPUT_P}/{title}.png", dpi=720)
plt.show()
plt.close()

# %%
