# %%
from __future__ import annotations

from dataclasses import dataclass
import itertools as itrtls
import logging
import os
from collections import defaultdict
from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import mylib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# %%
lgr = logging.getLogger()

# %%
OUTPUT_P = "outputs/"
os.makedirs(OUTPUT_P, exist_ok=True)

# %%
# pltcfg_p: str = "conf/accuracy_dropout.yaml"
# pltcfg_p: str = "conf/accuracy_dropout_noclf.yaml"
pltcfg_p: str = "conf/acc_dropout_noclfwxgbr.yaml"


# %%
@dataclass
class PlotConf:
    title: Optional[str]
    mktmpl_fn_name: str
    key: str
    expcfgs: list[ExpConf]


@dataclass
class ExpConf:
    subtitle: str
    exp_ps: dict[str, str]  # afa method to experiment path
    xlim: tuple[Optional[float], Optional[float]]
    ylim: tuple[Optional[float], Optional[float]]


# %%
keys_to_ylabel = {
    "acc": "accuracy",
    "f1-score": "f1-score",
}
method_to_label = {
    "make_templates_vanilla": "knn_rgr",
    "make_templates_fix_rounds": "knn_rgr",
    # "make_templates_reduce_features": "tafa-reduce-search",
    # "dagger-make_templates_vanilla": "greedy-dagger",
    # "dagger-make_templates_fix_rounds": "mutate-dagger",
    # "dagger-make_templates_reduce_features": "tafa-reduce-search-dagger",
    # "dropout-make_templates_vanilla": "greedy-dropout",
    # "dropout-make_templates_fix_rounds": "mutate-dropout",
}
order_fn = lambda k: {  # noqa: E731
    "knn_rgr": -1,
    "dropout-rgr": -2,
    "dropout-clf": -3,
    "dropout-xgbr": -4,
    "dagger-rgr": -5,
    "dagger-clf": -6,
}[k]
label_to_plot_kwargs = {
    "knn_rgr": {
        "color": "blue",
        "alpha": 0.7,
        "marker": ".",
    },
    "dropout-rgr": {
        "color": "red",
        "alpha": 0.7,
        "marker": ".",
    },
    "dropout-clf": {
        "color": "magenta",
        "alpha": 0.7,
        "marker": ".",
    },
    "dropout-xgbr": {
        "color": "indigo",
        "alpha": 0.7,
        "marker": ".",
    },
    "dagger-rgr": {
        "color": "green",
        "alpha": 0.7,
        "marker": ".",
    },
    "dagger-clf": {
        "color": "indigo",
        "alpha": 0.7,
        "marker": ".",
    },
    # "static": {
    #     "color": "orange",
    #     "alpha": 0.7,
    #     "marker": ".",
    # },
    # "make_templates_vanilla": {
    #     "color": "redorange",
    #     "alpha": 0.7,
    #     "marker": ".",
    # },
    # "dagger-make_templates_vanilla": {"color": "purple"},
    # "dagger-make_templates_fix_rounds": {"color": "dargreen"},
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


def load_tafa_metrics(exp_p: str, mktmpl_fn_name: str) -> list[pd.DataFrame]:
    # read source metrics from the mktmpl_exp or expmktmpl_run that an experiment points to
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        if not os.path.isdir(run_p):
            continue
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "mktmpl_config.yaml"))
        if mktmpl_fn_name != str.split(run_cfg.make_templates_fn._target_, ".")[-1]:
            continue
        run_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "config.yaml"))
        source_run_p: str
        if hasattr(run_cfg, "mktmpl_exp"):
            source_run_p: str = os.path.join(
                mylib.utils.get_project_root_dir(),
                run_cfg.mktmpl_exp.exp_p,
                str(run_cfg.mktmpl_exp.run_id),
            )
        elif hasattr(run_cfg, "mktmpl_run"):
            source_run_p = os.path.join(
                mylib.utils.get_project_root_dir(), run_cfg.mktmpl_run
            )
        else:
            lgr.warning(f"{run_p} points to source mktmpl run that does not exist")
            continue
        metrics_p = os.path.join(source_run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exits")
            continue
        # mktmpl_fn_name: str = str.split(run_cfg.make_templates_fn._target_, ".")[-1]
        # if (
        #     exclude_mktmplfn_name is not None
        #     and mktmpl_fn_name in exclude_mktmplfn_name
        # ):
        #     continue
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        if (
            "eval_val/feature observed" not in metrics_df
            and "eval/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue
        metrics_l.append(metrics_df)
    return metrics_l


# def load_tafa_dagger_metrics(
#     exp_p: str, exclude_mktmplfn_name: Optional[list[str]]
# ) -> dict[str, list[pd.DataFrame]]:
#     metrics_d: dict[str, list[pd.DataFrame]] = defaultdict(list)
#     for run_p in sorted(
#         os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
#         key=safe_str_to_int,
#     ):
#         run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
#         metrics_p: str = os.path.join(run_p, "metrics.csv")
#         if not os.path.exists(metrics_p):
#             lgr.warning(f"{metrics_p} does not exist")
#             continue
#         mktmpl_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "mktmpl_config.yaml"))
#         mktmpl_fn_name: str = str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]
#         if (
#             exclude_mktmplfn_name is not None
#             and mktmpl_fn_name in exclude_mktmplfn_name
#         ):
#             continue
#         metrics_df: pd.DataFrame = (
#             pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
#         )
#         if (
#             "eval_warmup/feature observed" not in metrics_df
#             and "eval_dagger/feature observed" not in metrics_df
#         ):
#             lgr.warning(f"eval_val/feature observed not in {metrics_p}")
#             continue
#         metrics_d[mktmpl_fn_name].append(metrics_df)
#     return metrics_d


def load_dropout_metrics(exp_p: str, mktmpl_fn_name: str) -> list[pd.DataFrame]:
    # load only dropout metrics
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        if not os.path.isdir(run_p):
            continue
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        mktmpl_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "mktmpl_config.yaml"))
        if mktmpl_fn_name != str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]:
            continue
        # mktmpl_fn_name: str = str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]
        # if (
        #     exclude_mktmplfn_name is not None
        #     and mktmpl_fn_name in exclude_mktmplfn_name
        # ):
        #     continue
        # load dropout metrics
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        if "eval/feature observed" not in metrics_df:
            lgr.warning(f"{metrics_p} does not contain eval/feature_observed")
            continue
        metrics_l.append(metrics_df)
    return metrics_l


def load_dagger_metrics(exp_p: str, mktmpl_fn_name: str) -> list[pd.DataFrame]:
    # load only dropout metrics
    metrics_l: list[pd.DataFrame] = list()
    for run_p in sorted(
        os.listdir(os.path.join(mylib.utils.get_project_root_dir(), exp_p)),
        key=safe_str_to_int,
    ):
        run_p: str = os.path.join(mylib.utils.get_project_root_dir(), exp_p, run_p)
        if not os.path.isdir(run_p):
            continue
        metrics_p: str = os.path.join(run_p, "metrics.csv")
        if not os.path.exists(metrics_p):
            lgr.warning(f"{metrics_p} does not exist")
            continue
        mktmpl_cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "mktmpl_config.yaml"))
        if mktmpl_fn_name != str.split(mktmpl_cfg.make_templates_fn._target_, ".")[-1]:
            continue
        # load dropout metrics
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        if "eval_dagger/feature observed" not in metrics_df:
            lgr.warning(f"{metrics_p} does not contain eval_dagger/feature_observed")
            continue
        metrics_l.append(metrics_df)
    return metrics_l


def load_metrics(
    mktmpl_fn_name: str, exp_ps: dict[str, str]
) -> dict[str, list[pd.DataFrame]]:
    metrics_d: dict[str, list[pd.DataFrame]] = dict()
    for name, exp_p in exp_ps.items():
        if str.split(name, "-")[0] == "dropout":
            metrics_d[name] = load_dropout_metrics(
                exp_p=exp_p, mktmpl_fn_name=mktmpl_fn_name
            )
        elif str.split(name, "-")[0] == "dagger":
            metrics_d[name] = load_dagger_metrics(
                exp_p=exp_p, mktmpl_fn_name=mktmpl_fn_name
            )
        if method_to_label[mktmpl_fn_name] not in metrics_d:
            metrics_d[method_to_label[mktmpl_fn_name]] = load_tafa_metrics(
                exp_p=exp_p, mktmpl_fn_name=mktmpl_fn_name
            )
    return metrics_d


def make_plots(
    pltcfg: PlotConf, metrics_dl: Optional[list[dict[str, list[pd.DataFrame]]]]
):
    if metrics_dl is None:
        metrics_dl = [
            load_metrics(mktmpl_fn_name=pltcfg.mktmpl_fn_name, exp_ps=_expcfg.exp_ps)
            for _expcfg in pltcfg.expcfgs
        ]
    metrics_dl = [
        {_k: _mdl[_k] for _k in sorted(_mdl, key=order_fn)} for _mdl in metrics_dl
    ]
    fig: Figure
    axs: list[list[Axes]]
    fig, axs = plt.subplots(
        nrows=1, ncols=len(pltcfg.expcfgs), squeeze=False, layout="constrained"
    )  # type:ignore
    label_set = list()
    line_set = list()
    for _expcfg, _ax, _metrics_d in zip(pltcfg.expcfgs, axs[0], metrics_dl):
        _ax.set_title(_expcfg.subtitle)
        _ax.set_box_aspect(1.0)
        for _name, _metrics_dfl in _metrics_d.items():
            _prefix: str
            if "eval/acc" in _metrics_dfl[0]:
                _prefix = "eval"
            elif "eval_val/acc" in _metrics_dfl[0]:
                _prefix = "eval_val"
            elif "eval_dagger/acc" in _metrics_dfl[0]:
                _prefix = "eval_dagger"
            else:
                logging.warning(f"{_name} has unsupported metrics key")
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
                **label_to_plot_kwargs[_name],
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
        _ylim[0] = _ylim[0] if _expcfg.ylim[0] is None else _expcfg.ylim[0]
        _ylim[1] = _ylim[1] if _expcfg.ylim[1] is None else _expcfg.ylim[1]
        _ax.set_ylim(*_ylim)
        # _ax.yaxis.set_major_formatter(
        #     FormatStrFormatter("%.3f")
        #     if abs(_ylim[1] - _ylim[0]) < 0.01
        #     else FormatStrFormatter("%.2f")
        # )
    return fig, axs, line_set, label_set, metrics_dl


# %%
pltcfg: PlotConf = OmegaConf.load(pltcfg_p)  # type:ignore

# %%
# metrics_dl = [
#     load_metrics(mktmpl_fn_name=pltcfg.mktmpl_fn_name, exp_ps=_expcfg.exp_ps)
#     for _expcfg in pltcfg.expcfgs
# ]
metrics_dl = None

# %%
fig, axs, line_set, label_set, metrics_dl = make_plots(
    pltcfg=pltcfg, metrics_dl=metrics_dl
)
fig.set_figheight(2.0 + 0.667 * max(len(label_set) // 4 - 1, 0))
fig.set_figwidth(1.7 * len(axs[0]))
if pltcfg.title is not None:
    fig.suptitle(pltcfg.title)
fig.supxlabel("number of feature acquisition\n")
# fig.supylabel(keys_to_ylabel[pltcfg.key])
axs[0][0].set_ylabel(keys_to_ylabel[pltcfg.key])
fig.legend(
    line_set,
    label_set,
    loc="outside lower center",
    frameon=False,
    ncol=6,
    # borderaxespad=1.0,
    # borderaxespad=0.18,
)
fn: str = os.path.splitext(pltcfg_p.split("/")[-1])[0]
fig.savefig(os.path.join(OUTPUT_P, f"{fn}.png"), dpi=720, bbox_inches="tight")
plt.show()
plt.close()

# %%
