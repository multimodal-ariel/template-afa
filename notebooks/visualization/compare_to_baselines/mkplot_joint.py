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
import pandas as pd
from omegaconf import OmegaConf

# %%
lgr = logging.getLogger()

# %%
OUTPUT_P = "outputs/joint"
os.makedirs(OUTPUT_P, exist_ok=True)

# %%
pltcfg_p: str = "conf/f1-score.yaml"


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
    xlim: tuple[Optional[float], Optional[float]]
    ylim: tuple[Optional[float], Optional[float]]


# %%
keys_to_ylabel = {
    "acc": "accuracy",
    "f1-score": "f1-score",
}
method_to_label = {
    "make_templates_vanilla": "greedy",
    "make_templates_fix_rounds": "mutate",
    # "make_templates_reduce_features": "tafa-reduce-search",
}


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
                metrics_d[method_to_label[_name]] = _metrics_l
    return metrics_d


def make_plots(pltcfg: PlotConf):
    fig: Figure
    axs: list[list[Axes]]
    fig, axs = plt.subplots(
        nrows=1, ncols=len(pltcfg.expcfgs), squeeze=False, layout="constrained"
    )  # type:ignore
    if pltcfg.title is not None:
        fig.suptitle(pltcfg.title)
    # fig.supxlabel("number of feature acquisition", y=0.1)
    fig.supxlabel("number of feature acquisition\n")
    fig.supylabel(keys_to_ylabel[pltcfg.key])
    label_set = list()
    line_set = list()
    for _expcfg, _ax in zip(pltcfg.expcfgs, axs[0]):
        _ax.set_title(_expcfg.subtitle)
        _ax.set_box_aspect(1.0)
        _metrics_d: dict[str, list[pd.DataFrame]] = load_metrics(
            _expcfg.exp_ps,
            load_kwargs={
                "tafa": {
                    "exclude_mktmplfn_name": [
                        "make_templates_reduce_features",
                        "make_templates_fix_rounds_nearest_neighbors",
                    ]
                }
            },
        )
        for _name, _metrics_dfl in _metrics_d.items():
            _prefix: str
            if "eval/acc" in _metrics_dfl[0]:
                _prefix = "eval"
            elif "eval_val/acc" in _metrics_dfl[0]:
                _prefix = "eval_val"
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
            _ax.plot(_data[:, 0], _data[:, 1], label=_name, marker="x")
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
    return fig, axs, line_set, label_set


# %%
pltcfg: PlotConf = OmegaConf.load(pltcfg_p)  # type:ignore

# %%
fig, axs, line_set, label_set = make_plots(pltcfg=pltcfg)
fig.set_figheight(2.0)
fig.set_figwidth(1.7 * len(axs[0]))
fig.legend(
    line_set,
    label_set,
    loc="outside lower center",
    frameon=False,
    ncol=len(line_set),
    borderaxespad=0.18,
)
fn: str = os.path.splitext(pltcfg_p.split("/")[-1])[0]
fig.savefig(os.path.join(OUTPUT_P, f"{fn}.png"), dpi=720, bbox_inches="tight")
plt.show()
plt.close()

# %%
