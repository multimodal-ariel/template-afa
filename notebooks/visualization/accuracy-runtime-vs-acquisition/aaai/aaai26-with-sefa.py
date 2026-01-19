# %%
from __future__ import annotations

import itertools as itrtls
import logging as lgr
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import matplotlib.pyplot as plt
import mylib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls",
    resolver=lambda cls: hd.utils.get_class(cls),
    replace=True,
)

# %%
pltcfg_p: str = "conf/accuracy_wjafa-wsefa.yaml"


# %%
@dataclass
class PlotConf:
    title: Optional[str]
    key: str
    expcfgs: list[ExpConf]


@dataclass
class ExpConf:
    subtitle: str
    exp_ps: dict[str, str]  # afa method to experiment path
    runtime_ps: dict[str, str]
    xlim: tuple[Optional[float], Optional[float]]
    ylim_row0: tuple[Optional[float], Optional[float]]


# %%
keys_to_ylabel = {
    "acc": "accuracy",
    "f1-score": "f1-score",
}
method_to_label = {
    "make_templates_vanilla": "greedy",
    "make_templates_fix_rounds": "tafa",
    # "tafa-dagger-dtc-make_templates_vanilla": "",
    "tafa-dagger-dtc-make_templates_fix_rounds": "tafa-interp.",
    # "make_templates_reduce_features": "tafa-reduce-search",
    # "dagger-make_templates_vanilla": "greedy-dagger",
    # "dagger-make_templates_fix_rounds": "mutate-dagger",
    # "dagger-make_templates_reduce_features": "tafa-reduce-search-dagger",
    # "dropout-make_templates_vanilla": "greedy-dropout",
    # "dropout-make_templates_fix_rounds": "mutate-dropout",
}
label_to_method = {
    "aco": "aco",
    "dime": "dime",
    "jafa": "jafa",
    "sefa": "sefa",
    "static": "static",
    "tafa-gumbel": "tafa-gumbel",
    "tafa-gumbel-knn": "tafa-gumbel-knn",
}
label_to_method.update({v: k for k, v in method_to_label.items()})
method_to_plot_kwargs = {
    "make_templates_fix_rounds": {
        "color": "red",
        "alpha": 0.7,
        "marker": ".",
    },
    "tafa-dagger-dtc-make_templates_fix_rounds": {
        "color": "blue",
        "alpha": 0.7,
        "marker": ".",
    },
    "aco": {
        "color": "cyan",
        "alpha": 0.7,
        "marker": ".",
    },
    "dime": {
        "color": "orange",
        "alpha": 0.7,
        "marker": ".",
    },
    "jafa": {
        "color": "indigo",
        "alpha": 0.7,
        "marker": ".",
    },
    "static": {
        "color": "green",
        "alpha": 0.7,
        "marker": ".",
    },
    "make_templates_vanilla": {
        "color": "redorange",
        "alpha": 0.7,
        "marker": ".",
    },
    "tafa-dagger-dtc-make_templates_vanilla": {
        "color": "purple",
        "alpha": 0.7,
        "marker": ".",
    },
    "sefa": {
        "color": "magenta",
        "alpha": 0.7,
        "marker": ".",
    },
    "tafa-gumbel": {
        "color": "lime",
        "alpha": 0.7,
        "marker": ".",
    },
    "tafa-gumbel-knn": {
        "color": "tan",
        "alpha": 0.7,
        "marker": ".",
    },
    # "dropout-make_templates_vanilla": {"color": "lime"},
    # "dropout-make_templates_fix_rounds": {"color": "cyan"},
}
# "color": "blue"
# "color": "red"
# "color": "purple"
# "color": "gold"
# "color": "darkgreen"
# "color": "lime"
# "color": "cyan"
# "color": "magenta"
# "color": "blueviolet"
# "color": "darkorange"
# "color": "tan"


def safe_str_to_int(s: str):
    try:
        return int(s)
    except ValueError as e:
        return -1


# %%
def load_baseline_metrics(exp_p: str) -> list[pd.DataFrame]:
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
        # print(run_p)
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        metrics_l.append(metrics_df)
    return metrics_l


def load_jafa_metrics(exp_p: str) -> list[pd.DataFrame]:
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "jafa_results.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        # print(run_p)
        metrics_df: pd.DataFrame = pd.read_csv(metrics_p)
        metrics_df.rename(
            columns={
                "val_acc": "eval/acc",
                "val_n_acquired_mean": "eval/feature observed",
            },
            inplace=True,
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
        # print(run_p)
        metrics_d[mktmpl_fn_name].append(metrics_df)
    return metrics_d


def load_tafa_dagger_metrics(
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
        mktmpl_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "mktmpl_config.yaml"))
        mktmpl_fn_name: str = str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]
        if (
            exclude_mktmplfn_name is not None
            and mktmpl_fn_name in exclude_mktmplfn_name
        ):
            continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        if (
            "eval_warmup/feature observed" not in metrics_df
            and "eval_dagger/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue
        # print(run_p)
        metrics_d[mktmpl_fn_name].append(metrics_df)
    return metrics_d


def load_tafa_dagger_dtc_metrics(
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
        mktmpl_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
        mktmpl_cfg = OmegaConf.load(
            os.path.join(
                mylib.utils.get_project_root_dir(),
                mktmpl_cfg.train_exp.exp_p,
                str(mktmpl_cfg.train_exp.run_id),
                ".hydra",
                "config.yaml",
            )
        )
        mktmpl_fn_name: str = str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]
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
        # print(run_p)
        metrics_d[mktmpl_fn_name].append(metrics_df)
    return metrics_d


def load_tafa_dropout_metrics(
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
        mktmpl_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "mktmpl_config.yaml"))
        mktmpl_fn_name: str = str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]
        if (
            exclude_mktmplfn_name is not None
            and mktmpl_fn_name in exclude_mktmplfn_name
        ):
            continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        if "eval/feature observed" not in metrics_df:
            continue
        # print(run_p)
        metrics_d[mktmpl_fn_name].append(metrics_df)
    return metrics_d


def load_sefa_metrics(csv_path: str) -> dict[str, list[pd.DataFrame]]:
    """Load SEFA results from CSV file."""
    if not csv_path:  # Empty string case
        return {"sefa": []}

    full_path = os.path.join(mylib.utils.get_project_root_dir(), csv_path)

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        # Map columns to expected format
        df_formatted = pd.DataFrame(
            {"eval/feature observed": df["test_cost"], "eval/acc": df["test_acc"]}
        )
        return {"sefa": [df_formatted]}
    else:
        lgr.warning(f"SEFA results file not found at {full_path}")
        return {"sefa": []}


def load_tafa_gumbel_metrics(csv_path: str) -> dict[str, list[pd.DataFrame]]:
    """Load Gumbel dropout ensemble results from CSV file."""
    if not csv_path:  # Empty string case
        return {"tafa-gumbel": []}

    full_path = os.path.join(mylib.utils.get_project_root_dir(), csv_path)

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        # Map columns to expected format
        df_formatted = pd.DataFrame(
            {"eval/feature observed": df["test_cost"], "eval/acc": df["test_acc"]}
        )
        return {"tafa-gumbel": [df_formatted]}
    else:
        lgr.warning(f"Gumbel results file not found at {full_path}")
        return {"tafa-gumbel": []}


def load_tafa_gumbel_knn_metrics(exp_p: str) -> list[pd.DataFrame]:
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p))
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
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
        # print(run_p)
        metrics_l.append(metrics_df)
    return metrics_l


def load_metrics(
    dataset_name: str, exp_ps: dict[str, str], load_kwargs: Optional[dict[str, Any]]
) -> dict[str, list[pd.DataFrame]]:
    metrics_d: dict[str, list[pd.DataFrame]] = dict()
    for name, exp_p in exp_ps.items():
        if name in {"aco", "dime", "static"}:
            metrics_d[name] = load_baseline_metrics(exp_p)
        elif name == "jafa":
            metrics_d[name] = load_jafa_metrics(exp_p)
        elif name == "tafa":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_metrics(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[method_to_label[_name]] = _metrics_l
        elif name == "tafa-dagger-dtc":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_dagger_dtc_metrics(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[method_to_label[f"tafa-dagger-dtc-{_name}"]] = _metrics_l
        elif name == "tafa-dropout":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa-dagger"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_dropout_metrics(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[method_to_label[f"dropout-{_name}"]] = _metrics_l
        elif name == "tafa-dagger":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa-dagger"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_dagger_metrics(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[method_to_label[f"dagger-{_name}"]] = _metrics_l
        elif name == "tafa-gumbel":
            # NOTE exp_p is now the CSV path
            gumbel_results = load_tafa_gumbel_metrics(exp_p)
            for method_name, metrics_list in gumbel_results.items():
                metrics_d[method_name] = metrics_list
        elif name == "tafa-gumbel-knn":
            metrics_d[name] = load_tafa_gumbel_knn_metrics(exp_p)
        elif name == "sefa":
            sefa_results = load_sefa_metrics(exp_p)
            for method_name, metrics_list in sefa_results.items():
                metrics_d[method_name] = metrics_list
    return metrics_d


# %%
def load_baseline_runtime(exp_p: str) -> pd.DataFrame:
    exp_p = os.path.join(mylib.utils.get_project_root_dir(), exp_p)
    metrics_df = pd.concat(
        [
            pd.read_csv(os.path.join(exp_p, run_p, "metrics.csv"))
            for run_p in os.listdir(exp_p)
            if os.path.isdir(os.path.join(exp_p, run_p))
        ]
    ).sort_values("eval/feature observed")
    # plt.plot(metrics_df["eval/feature observed"], metrics_df["eval/avg_pred_time_ns"])
    return metrics_df


def load_jafa_runtime(exp_p: str) -> pd.DataFrame:
    exp_p = os.path.join(mylib.utils.get_project_root_dir(), exp_p)
    metrics_df = pd.concat(
        [
            pd.read_csv(os.path.join(exp_p, run_p, "metrics.csv"))
            for run_p in os.listdir(exp_p)
            if os.path.isdir(os.path.join(exp_p, run_p))
            and os.path.exists(os.path.join(exp_p, run_p, "metrics.csv"))
        ]
    ).sort_values("val_n_acquired_mean")
    metrics_df.rename(
        columns={
            "val_n_acquired_mean": "eval/feature observed",
            "val_avg_pred_time_ns": "eval/avg_pred_time_ns",
        },
        inplace=True,
    )
    # plt.plot(metrics_df["val_n_acquired_mean"], metrics_df["val_avg_pred_time_ns"])
    return metrics_df


def load_tafa_runtime(
    exp_p: str, exclude_mktmplfn_name: Optional[list[str]]
) -> dict[str, pd.DataFrame]:
    _metrics_d: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        # load the runtime config
        run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
        # fish out original mktmpl config
        run_cfg = OmegaConf.load(
            os.path.join(
                mylib.utils.get_project_root_dir(),
                run_cfg.train_exp.exp_p,
                str(run_cfg.train_exp.run_id),
                ".hydra",
                "config.yaml",
            )
        )
        mktmpl_fn_name: str = str.split(run_cfg.make_templates_fn._target_, ".")[-1]
        if (
            exclude_mktmplfn_name is not None
            and mktmpl_fn_name in exclude_mktmplfn_name
        ):
            continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p)
            .groupby("step")
            .sum(min_count=1)
            .sort_values("eval/feature observed")
        )
        if (
            "eval_val/feature observed" not in metrics_df
            and "eval/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue
        # print(run_p)
        _metrics_d[mktmpl_fn_name].append(metrics_df)
    metrics_d: dict[str, pd.DataFrame] = {
        k: pd.concat(v).sort_values("eval/feature observed")
        for k, v in _metrics_d.items()
    }
    return metrics_d


def load_tafa_dagger_dtc_runtime(
    exp_p: str, exclude_mktmplfn_name: Optional[list[str]]
) -> dict[str, pd.DataFrame]:
    _metrics_d: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        # load the runtime config
        run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
        # fish out original mktmpl config
        run_cfg = OmegaConf.load(
            os.path.join(
                mylib.utils.get_project_root_dir(),
                run_cfg.train_exp.exp_p,
                str(run_cfg.train_exp.run_id),
                ".hydra",
                "config.yaml",
            )
        )
        mktmpl_fn_name: str = str.split(run_cfg.make_templates_fn._target_, ".")[-1]
        if (
            exclude_mktmplfn_name is not None
            and mktmpl_fn_name in exclude_mktmplfn_name
        ):
            continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p)
            .groupby("step")
            .sum(min_count=1)
            .sort_values("eval/feature observed")
        )
        if (
            "eval_val/feature observed" not in metrics_df
            and "eval/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue
        # print(run_p)
        _metrics_d[mktmpl_fn_name].append(metrics_df)
    metrics_d: dict[str, pd.DataFrame] = {
        k: pd.concat(v).sort_values("eval/feature observed")
        for k, v in _metrics_d.items()
    }
    return metrics_d


def load_tafa_gumbel_runtime(csv_path: str) -> pd.DataFrame:
    """Load Gumbel dropout ensemble runtime results from CSV file."""
    if not csv_path:  # Empty string case
        return pd.DataFrame()

    full_path = os.path.join(mylib.utils.get_project_root_dir(), csv_path)

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        # Map columns to expected format and sort by feature observed
        df_formatted = pd.DataFrame(
            {
                "eval/feature observed": df["test_cost"],
                "eval/avg_pred_time_ns": df["time_per_sample_ns"],
            }
        ).sort_values("eval/feature observed")
        return df_formatted
    else:
        lgr.warning(f"Gumbel runtime file not found at {full_path}")
        return pd.DataFrame()


def load_tafa_gumbel_knn_runtime(exp_p: str) -> pd.DataFrame:
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p))
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
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
        # print(run_p)
        metrics_l.append(metrics_df)
    metrics_df = pd.concat(metrics_l).sort_values("eval/feature observed")
    return metrics_df


def load_sefa_runtime(csv_path: str) -> pd.DataFrame:
    """Load SEFA runtime results from CSV file."""
    if not csv_path:  # Empty string case
        return pd.DataFrame()

    full_path = os.path.join(mylib.utils.get_project_root_dir(), csv_path)

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        # Map columns to expected format and sort by feature observed
        df_formatted = pd.DataFrame(
            {
                "eval/feature observed": df["test_cost"],
                "eval/avg_pred_time_ns": df["cumulative_time_ns"],
            }
        ).sort_values("eval/feature observed")
        return df_formatted
    else:
        lgr.warning(f"SEFA runtime file not found at {full_path}")
        return pd.DataFrame()


def load_runtimes(
    exp_ps: dict[str, str], load_kwargs: Optional[dict[str, Any]]
) -> dict[str, pd.DataFrame]:
    metrics_d: dict[str, pd.DataFrame] = dict()
    for name, exp_p in exp_ps.items():
        if name in {"aco", "dime", "static"}:
            metrics_d[name] = load_baseline_runtime(exp_p)
        elif name == "jafa":
            metrics_d[name] = load_jafa_runtime(exp_p)
        elif name == "tafa":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_runtime(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[method_to_label[_name]] = _metrics_l
        elif name == "tafa-dagger-dtc":
            _exclude_mktmplfn_name = (
                load_kwargs["tafa"]["exclude_mktmplfn_name"]
                if load_kwargs is not None and "tafa" in load_kwargs
                else None
            )
            for _name, _metrics_l in load_tafa_runtime(
                exp_p, _exclude_mktmplfn_name
            ).items():
                metrics_d[method_to_label[f"tafa-dagger-dtc-{_name}"]] = _metrics_l
        elif name == "tafa-gumbel":
            # NOTE exp_p is now path to csv
            metrics_d["tafa-gumbel"] = load_tafa_gumbel_runtime(exp_p)
        elif name == "tafa-gumbel-knn":
            metrics_d[name] = load_tafa_gumbel_knn_runtime(exp_p)
        elif name == "sefa":
            metrics_d["sefa"] = load_sefa_runtime(exp_p)
        # elif name == "tafa-dropout":
        #     _exclude_mktmplfn_name = (
        #         load_kwargs["tafa-dagger"]["exclude_mktmplfn_name"]
        #         if load_kwargs is not None and "tafa" in load_kwargs
        #         else None
        #     )
        #     for _name, _metrics_l in load_tafa_dropout_metrics(
        #         exp_p, _exclude_mktmplfn_name
        #     ).items():
        #         metrics_d[method_to_label[f"dropout-{_name}"]] = _metrics_l
        # elif name == "tafa-dagger":
        #     _exclude_mktmplfn_name = (
        #         load_kwargs["tafa-dagger"]["exclude_mktmplfn_name"]
        #         if load_kwargs is not None and "tafa" in load_kwargs
        #         else None
        #     )
        #     for _name, _metrics_l in load_tafa_dagger_metrics(
        #         exp_p, _exclude_mktmplfn_name
        #     ).items():
        #         metrics_d[method_to_label[f"dagger-{_name}"]] = _metrics_l
    return metrics_d


# %%
def make_plots(
    pltcfg: PlotConf,
    metrics_dl: Optional[list[dict[str, list[pd.DataFrame]]]],
    runtimes_dl: Optional[list[dict[str, pd.DataFrame]]],
):
    if metrics_dl is None:
        metrics_dl = [
            load_metrics(
                _expcfg.subtitle,
                _expcfg.exp_ps,
                # {**_expcfg.exp_ps, "sefa": ""},
                load_kwargs={
                    "tafa": {
                        "exclude_mktmplfn_name": [
                            "make_templates_reduce_features",
                            "make_templates_fix_rounds_nearest_neighbors",
                            "make_templates_vanilla",
                        ]
                    },
                    "tafa-dagger-dtc": {
                        "exclude_mktmplfn_name": [
                            "make_templates_reduce_features",
                            "make_templates_fix_rounds_nearest_neighbors",
                            "make_templates_vanilla",
                        ]
                    },
                },
            )
            for _expcfg in pltcfg.expcfgs
        ]
    if runtimes_dl is None:
        runtimes_dl = [
            load_runtimes(
                _expcfg.runtime_ps,
                load_kwargs={
                    "tafa": {
                        "exclude_mktmplfn_name": [
                            "make_templates_reduce_features",
                            "make_templates_fix_rounds_nearest_neighbors",
                            "make_templates_vanilla",
                        ]
                    },
                    "tafa-dagger-dtc": {
                        "exclude_mktmplfn_name": [
                            "make_templates_reduce_features",
                            "make_templates_fix_rounds_nearest_neighbors",
                            "make_templates_vanilla",
                        ]
                    },
                },
            )
            for _expcfg in pltcfg.expcfgs
        ]
    fig: Figure
    axs: list[list[Axes]]
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(pltcfg.expcfgs),
        squeeze=False,
        layout="constrained",
        sharex="col",
    )  # type:ignore
    label_set = list()
    line_set = list()
    for _expcfg, _ax, _metrics_d in zip(pltcfg.expcfgs, axs[0], metrics_dl):
        _ax.set_title(_expcfg.subtitle, fontsize="xx-large")
        _ax.set_box_aspect(1.0)
        for _name, _metrics_dfl in reversed(_metrics_d.items()):
            if len(_metrics_dfl) == 0:
                continue
            _prefix: str
            if "eval/acc" in _metrics_dfl[0]:
                _prefix = "eval"
            elif "eval_val/acc" in _metrics_dfl[0]:
                _prefix = "eval_val"
            elif "eval_dagger/acc" in _metrics_dfl[0]:
                _prefix = "eval_dagger"
            else:
                lgr.warning(f"{_name} has unsupported metrics key")
                continue
            _data = pd.concat(
                [
                    _mdf[
                        [f"{_prefix}/feature observed", f"{_prefix}/{pltcfg.key}"]
                    ].dropna()
                    for _mdf in _metrics_dfl
                ]
            ).to_numpy()
            _data = _data[np.argsort(_data[:, 0]).flatten()]
            _ax.plot(
                _data[:, 0],
                _data[:, 1],
                label=_name,
                # marker="x",
                **method_to_plot_kwargs[label_to_method[_name]],
            )
        _ax_lines, _ax_labels = _ax.get_legend_handles_labels()
        for _ax_line, _ax_label in zip(_ax_lines, _ax_labels):
            if _ax_label not in label_set:
                line_set.append(_ax_line)
                label_set.append(_ax_label)
        _xlim, _ylim = [*_ax.get_xlim()], [*_ax.get_ylim()]
        _xlim[0] = _xlim[0] if _expcfg.xlim[0] is None else _expcfg.xlim[0]
        _xlim[1] = _xlim[1] if _expcfg.xlim[1] is None else _expcfg.xlim[1]
        _ax.set_xlim(*_xlim)
        _ylim[0] = _ylim[0] if _expcfg.ylim_row0[0] is None else _expcfg.ylim_row0[0]
        _ylim[1] = _ylim[1] if _expcfg.ylim_row0[1] is None else _expcfg.ylim_row0[1]
        _ax.set_ylim(*_ylim)
        # _ax.yaxis.set_major_formatter(
        #     FormatStrFormatter("%.3f")
        #     if abs(_ylim[1] - _ylim[0]) < 0.01
        #     else FormatStrFormatter("%.2f")
        # )
    for _expcfg, _ax, _runtimes_d in zip(pltcfg.expcfgs, axs[1], runtimes_dl):
        # _ax.set_title(_expcfg.subtitle)
        _ax.set_box_aspect(1.0)
        for _name, _runtimes_df in reversed(_runtimes_d.items()):
            if len(_runtimes_df) == 0:
                continue
            _ax.plot(
                _runtimes_df["eval/feature observed"],
                _runtimes_df["eval/avg_pred_time_ns"] * 1e-9,
                label=_name,
                **method_to_plot_kwargs[label_to_method[_name]],
            )
        _ax_lines, _ax_labels = _ax.get_legend_handles_labels()
        for _ax_line, _ax_label in zip(_ax_lines, _ax_labels):
            if _ax_label not in label_set:
                line_set.append(_ax_line)
                label_set.append(_ax_label)
        _xlim, _ylim = [*_ax.get_xlim()], [*_ax.get_ylim()]
        _xlim[0] = _xlim[0] if _expcfg.xlim[0] is None else _expcfg.xlim[0]
        _xlim[1] = _xlim[1] if _expcfg.xlim[1] is None else _expcfg.xlim[1]
        _ax.set_xlim(*_xlim)
        _ylim[0] = _ylim[0] if _expcfg.ylim_row0[0] is None else _expcfg.ylim_row0[0]
        _ylim[1] = _ylim[1] if _expcfg.ylim_row0[1] is None else _expcfg.ylim_row0[1]
        # _ax.set_ylim(*_ylim)
        _ax.set_yscale("log")
    return fig, axs, line_set, label_set, metrics_dl, runtimes_dl


# %%
pltcfg: PlotConf = OmegaConf.load(pltcfg_p)  # type:ignore

# %%
metrics_dl = None
runtimes_dl = None

# %%
fig, axs, line_set, label_set, metrics_dl, runtimes_dl = make_plots(
    pltcfg=pltcfg, metrics_dl=metrics_dl, runtimes_dl=runtimes_dl
)
fig.set_figheight(1.75 * 2.0 + 0.667 * max(len(label_set) // 4 - 1, 0))
fig.set_figwidth(1.85 * len(axs[0]))
if pltcfg.title is not None:
    fig.suptitle(pltcfg.title)
fig.supxlabel("number of features acquired\n", fontsize="x-large")
# fig.supylabel(keys_to_ylabel[pltcfg.key])
axs[0][0].set_ylabel(keys_to_ylabel[pltcfg.key], fontsize="large")
axs[1][0].set_ylabel("log eval time [s]", fontsize="large")
fig.legend(
    list(reversed(line_set)),
    list(reversed(label_set)),
    loc="outside lower center",
    frameon=False,
    ncol=10,
    # ncol=6,
    fontsize="large",
    # borderaxespad=1.0,
    # borderaxespad=0.18,
)
fn: str = os.path.splitext(pltcfg_p.split("/")[-1])[0]
os.makedirs("outputs", exist_ok=True)
fig.savefig(os.path.join("outputs", f"{fn}.png"), dpi=720, bbox_inches="tight")
plt.show()
plt.close()

# %%
