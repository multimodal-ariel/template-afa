# %%
import logging as logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import mylib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from omegaconf import OmegaConf

# %%
lgr = logging.getLogger()


# %%
exp_ps_l: list[str] = [
    "experiments/make_template/outputs/cube_ncands_sweep/20250331_025659",
    "experiments/make_template/outputs/cube_ncands_sweep/20250331_095250",
    "experiments/make_template/outputs/cube_ncands_sweep_intra-corr/20250713_193214",
]

# %%
outputs_p: str = "outputs"
os.makedirs(outputs_p, exist_ok=True)

# %%
method_to_label = {
    "make_templates_fix_rounds": "mutate",
    "make_templates_vanilla": "practical",
}
method_to_plot_kwargs = {
    "make_templates_fix_rounds": {
        "color": "blue",
        # "alpha": 0.7,
        # "marker": "|",
    },
    "make_templates_vanilla": {
        "color": "red",
        # "alpha": 0.7,
        # "marker": "|",
    },
    # "aco": {
    #     "color": "darkorange",
    #     "alpha": 0.7,
    #     "marker": "|",
    # },
    # "dime": {
    #     "color": "magenta",
    #     "alpha": 0.7,
    #     "marker": ".",
    # },
    # "jafa": {
    #     "color": "darkgreen",
    #     "alpha": 0.7,
    #     "marker": "|",
    # },
    # "static": {
    #     "color": "purple",
    #     "alpha": 0.7,
    #     "marker": "|",
    # },
    # "dagger-make_templates_vanilla": {"color": "purple"},
    # "dagger-make_templates_fix_rounds": {"color": "dargreen"},
    # "dropout-make_templates_vanilla": {"color": "lime"},
    # "dropout-make_templates_fix_rounds": {"color": "cyan"},
}


def safe_str_to_int(s: str):
    try:
        return int(s)
    except ValueError as e:
        return -1


def load_tafa_metrics(exp_p: str) -> dict[str, list[pd.DataFrame]]:
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
        metrics_df: pd.DataFrame = (
            pd.read_csv(metrics_p).groupby("step").sum(min_count=1)
        )
        _prefix: str
        if "eval/acc" in metrics_df:
            _prefix = "eval"
        elif "eval_val/acc" in metrics_df:
            _prefix = "eval_val"
        elif "eval_dagger/acc" in metrics_df:
            _prefix = "eval_dagger"
        else:
            logging.warning(f"{metrics_p} has unsupported metrics key")
            continue
        metrics_df[f"{_prefix}/n_cands"] = (
            run_cfg.make_templates_fn.n_cands
            if hasattr(run_cfg.make_templates_fn, "n_cands")
            else run_cfg.make_templates_fn.n_cands_init
            * run_cfg.make_templates_fn.n_rounds
        )
        if (
            "eval_val/feature observed" not in metrics_df
            and "eval/feature observed" not in metrics_df
        ):
            lgr.warning(f"eval_val/feature observed not in {metrics_p}")
            continue
        metrics_d[mktmpl_fn_name].append(metrics_df)
    return metrics_d


# %%
metrics_df: dict[str, list[pd.DataFrame]] = dict()
for _p in exp_ps_l:
    metrics_df.update(load_tafa_metrics(_p))

# %%
fig: Figure
ax: Axes
fig, ax = plt.subplots(layout="constrained")
# fig.tight_layout()
fig.set_figheight(2.3)
fig.set_figwidth(2.6)
for _n, _mdfsl in metrics_df.items():
    _data = pd.concat(
        [_mdf[["eval/n_cands", "eval/acc"]].dropna() for _mdf in _mdfsl]
    ).to_numpy()
    _data = _data[np.argsort(_data[:, 0]).flatten()]
    if _n not in method_to_plot_kwargs:
        continue
    ax.plot(
        _data[:, 0], _data[:, 1], label=method_to_label[_n], **method_to_plot_kwargs[_n]
    )
# fig.set_figwidth(0.6 + len(labels_l) * 0.4)
# ax.bar(ind, ts_aaco, width, label="aaco")
# ax.bar(ind + width, ts_tafa, width, label="ours")
# ax.set_xticks(ind + width / 2, labels_l)
mkformatter = FuncFormatter(
    lambda x, _: (
        "%1.1fM" % (x * 1e-6)
        if x >= 1e6
        else "%1.1fK" % (x * 1e-3) if x >= 1e3 else "%1.1f" % x
    )
)
ax.xaxis.set_major_formatter(mkformatter)
ax.set_title("cube")
ax.set_ylim(0.65, 0.765)
ax.set_ylabel("accuracy")
ax.set_xlim(0.0, 10_000)
ax.set_xlabel("candidate set size")
ax.legend(
    # ncols=2,
    # loc="outside lower center",
    frameon=False,
)
# fig.savefig(os.path.join(outputs_p, "acc_vs_ncands.png"), dpi=720)
plt.show()

# %%
