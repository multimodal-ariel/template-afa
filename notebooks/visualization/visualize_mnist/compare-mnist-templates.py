# %%
from __future__ import annotations

import itertools as itrtls
import os

import matplotlib.pyplot as plt
import mydatasets
import mylib
import tensordict as thd
import torch as th
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure

# %%
run_p: str = "experiments/make_template/outputs/mnist_cnnet/20250326_003820/0"
run_p = os.path.join(mylib.utils.get_project_root_dir(), run_p)
out_p: str = "outputs/visualize-mnist"
os.makedirs(out_p, exist_ok=True)

# %%
# keys:
#   - "xs": (n, n_covs) for mnist, n_covs = 16*16 = 256
#   - "ys": (n, )
_tdata: thd.TensorDict
vdata: thd.TensorDict
tstdata: thd.TensorDict
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("mnist", to_normalize=False)
_tdata_shuffle_idxs: th.Tensor = th.load(
    os.path.join(run_p, "tdata_shuffle_idxs.pt"), weights_only=False
)
tdata: thd.TensorDict = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
# (n_tmpls, n_covs)
tmpls: th.Tensor = th.load(os.path.join(run_p, "tmpls.pt"), weights_only=False)
# (len(tdata), )
# keys:
#   - "cels": (len(tdata), n_tmpls)
#   - "pyhats": (len(tdata), n_tmpls, n_labels)
#   - "rwds": (len(tdata), n_tmpls)
tpcomp: thd.TensorDict = th.load(os.path.join(run_p, "tpcomp.pt"), weights_only=False)

# %%
to_vis_tids: list[int] = (
    th.sort(th.bincount(th.max(tpcomp["rwds"], dim=1).indices), descending=True)
    # .indices[:4]
    # .indices[[0, 3, 5, 7]]
    .indices[[0, 2, 6]].tolist()
)

# %%
n_tovis: int = len(to_vis_tids)
n_tops: int = 4
n_leasts: int = 0
# make figure
fig = plt.figure(layout="compressed")
subfigs: list[SubFigure]
if n_leasts != 0:
    subfigs = fig.subfigures(1, 4, squeeze=True, width_ratios=(1, 1, n_tops, n_leasts))
else:
    subfigs = fig.subfigures(1, 3, squeeze=True, width_ratios=(1, 1, n_tops))
fig.set_figwidth(1.8 * (2 + n_tops + n_leasts) + 1.8)
fig.set_figheight(2 * n_tovis)
# make subplots
subfigs[0].suptitle("label dist.", x=0.65, fontsize=23)
subfigs[0].supylabel("num. instances", fontsize=19)
# subfigs[0].supxlabel("label", y=0.006)
# subfigs[0].supxlabel("label", y=.01, ha="left", x=0)
lbldists_axs: list[list[Axes]] = subfigs[0].subplots(n_tovis, 1, squeeze=False)
subfigs[1].suptitle("template", x=0.55, fontsize=23)
tmpl_axs: list[list[Axes]] = subfigs[1].subplots(n_tovis, 1, squeeze=False)
subfigs[2].suptitle("example instances utilizing template", fontsize=23)
top_axs: list[list[Axes]] = subfigs[2].subplots(n_tovis, n_tops, squeeze=False)
top_indices: tuple[tuple[int | None, ...], ...] = (
    (None, None, None, None),
    (None, None, None, None),
    (None, None, None, None),
    (None, None, None, None),
)
least_axs: list[list[Axes | None]]
if n_leasts != 0:
    subfigs[3].suptitle("least likely")
    least_axs = subfigs[3].subplots(n_tovis, n_leasts, squeeze=False)
else:
    least_axs = [list() for _ in range(n_tovis)]
for _tid, _tmpl_axs, _lbldists_axs, _top_axs, _top_idxs, _least_axs in zip(
    to_vis_tids, tmpl_axs, lbldists_axs, top_axs, top_indices, least_axs
):
    for _ax in itrtls.chain(
        _tmpl_axs,
        _lbldists_axs,
        _top_axs,
        _least_axs,
    ):
        _ax.set_box_aspect(1.0)
    # for _ax in itrtls.chain(_tmpl_axs, _top_axs, _least_axs):
    #     _ax.xaxis.set_visible(False)
    #     _ax.yaxis.set_visible(False)
    # index of instances using current template
    _idxs: th.Tensor = th.max(tpcomp["rwds"], dim=1).indices == _tid
    # plot current template
    _tmpl: th.Tensor = th.reshape(tmpls[_tid], (16, 16))
    # _tmpl_axs[0].set_title(f"tmpl. {_tid}")
    _tmpl_axs[0].imshow(
        th.stack([_tmpl * 255 for _ in range(3)], dim=-1), cmap="copper"
    )
    # plot label distribution
    # class label for instances using ucrrent template
    _labels: th.Tensor = tdata["ys"][_idxs]
    # plot label distribution
    _lbl_dist: th.Tensor = th.bincount(_labels, minlength=n_labels)
    _lbldists_axs[0].bar(range(n_labels), _lbl_dist)
    _lbldists_axs[0].set_box_aspect(1.18)
    # _lbldists_axs[0].set_title(f"{_tid}-th tmpl.")
    # _lbldists_axs[0].set_ylabel("num. instances")
    _lbldists_axs[0].set_xticks([1, 3, 5, 7, 9])
    # plot top 3 labels
    _top_labels: list[int] = th.argsort(_lbl_dist, descending=True).tolist()[:n_tops]
    for _i, (_lbl, _top_idx) in enumerate(zip(_top_labels, _top_idxs)):
        _ax: Axes = _top_axs[_i]
        _tdata: thd.TensorDict = tdata[tdata["ys"] == _lbl]
        _img: th.Tensor = th.reshape(
            _tdata["xs"][
                0 if _top_idx is None else int(th.randint(0, len(_tdata), ()).item())
            ],
            (16, 16),
        )
        _ax.imshow(th.stack([_img for _ in range(3)], dim=-1))
        _pxl_locs: th.Tensor = th.argwhere(_tmpl == 1)
        _ax.scatter(
            x=_pxl_locs[:, 1],
            y=_pxl_locs[:, 0],
            marker="s",
            color="gold",
            alpha=0.7,
            edgecolors="none",
        )
    # plot two least likely label
    if n_leasts == 0:
        continue
    _least_label: list[int] = th.argsort(_lbl_dist, descending=False).tolist()[
        :n_leasts
    ]
    for _i, _lbl in enumerate(_least_label):
        _ax: Axes = _least_axs[_i]
        _tdata: thd.TensorDict = tdata[tdata["ys"] == _lbl]
        _img: th.Tensor = th.reshape(_tdata["xs"][_i], (16, 16))
        _ax.imshow(th.stack([_img for _ in range(3)], dim=-1))
        _pxl_locs: th.Tensor = th.argwhere(_tmpl == 1)
        _ax.scatter(
            x=_pxl_locs[:, 1],
            y=_pxl_locs[:, 0],
            marker="s",
            color="gold",
            alpha=0.7,
            edgecolors="none",
        )
fig.savefig(
    os.path.join(out_p, "mnist-tmpl.png"),
    dpi=720,
    bbox_inches="tight",
)
plt.show()
plt.close()

# %%
# n_tovis: int = len(to_vis_tids)
# n_tops: int = 4
# n_leasts: int = 0
# # make figure
# fig = plt.figure(layout="compressed")
# subfigs: list[SubFigure]
# if n_leasts != 0:
#     subfigs = fig.subfigures(1, 4, squeeze=True, width_ratios=(1, 1, n_tops, n_leasts))
# else:
#     subfigs = fig.subfigures(1, 3, squeeze=True, width_ratios=(1, 1, n_tops))
# fig.set_figwidth(1.8 * (2 + n_tops + n_leasts))
# fig.set_figheight(2 * n_tovis)
# # make subplots
# subfigs[0].suptitle("label dist.", x=0.65)
# # subfigs[0].supxlabel("label", y=0.006)
# # subfigs[0].supxlabel("label", y=.01, ha="left", x=0)
# lbldists_axs: list[list[Axes]] = subfigs[0].subplots(n_tovis, 1, squeeze=False)
# subfigs[1].suptitle("template", x=0.55)
# tmpl_axs: list[list[Axes]] = subfigs[1].subplots(n_tovis, 1, squeeze=False)
# subfigs[2].suptitle("example instances utilizing template")
# top_axs: list[list[Axes]] = subfigs[2].subplots(n_tovis, n_tops, squeeze=False)
# least_axs: list[list[Axes]] | None
# if n_leasts != 0:
#     subfigs[3].suptitle("least likely")
#     least_axs = subfigs[3].subplots(n_tovis, n_leasts, squeeze=False)
# else:
#     least_axs = [list() for _ in range(n_tovis)]
# for (
#     _tid,
#     _tmpl_axs,
#     _lbldists_axs,
#     _top_axs,
#     _least_axs,
# ) in zip(to_vis_tids, tmpl_axs, lbldists_axs, top_axs, least_axs):
#     for _ax in itrtls.chain(_tmpl_axs, _lbldists_axs, _top_axs, _least_axs):
#         _ax.set_box_aspect(1.0)
#     # for _ax in itrtls.chain(_tmpl_axs, _top_axs, _least_axs):
#     #     _ax.xaxis.set_visible(False)
#     #     _ax.yaxis.set_visible(False)
#     # index of instances using current template
#     _idxs: th.Tensor = th.max(tpcomp["rwds"], dim=1).indices == _tid
#     # plot current template
#     _tmpl: th.Tensor = th.reshape(tmpls[_tid], (16, 16))
#     # _tmpl_axs[0].set_title(f"tmpl. {_tid}")
#     _tmpl_axs[0].imshow(
#         th.stack([_tmpl * 255 for _ in range(3)], dim=-1), cmap="copper"
#     )
#     # plot label distribution
#     # class label for instances using ucrrent template
#     _labels: th.Tensor = tdata["ys"][_idxs]
#     # plot label distribution
#     _lbl_dist: th.Tensor = th.bincount(_labels, minlength=n_labels)
#     _lbldists_axs[0].bar(range(n_labels), _lbl_dist)
#     _lbldists_axs[0].set_box_aspect(1.18)
#     # _lbldists_axs[0].set_title(f"{_tid}-th tmpl.")
#     _lbldists_axs[0].set_ylabel("num. instances")
#     _lbldists_axs[0].set_xticks([1, 3, 5, 7, 9])
#     # plot top 3 labels
#     _top_labels: list[int] = th.argsort(_lbl_dist, descending=True).tolist()[:n_tops]
#     for _i, _lbl in enumerate(_top_labels):
#         _ax: Axes = _top_axs[_i]
#         _tdata: thd.TensorDict = tdata[tdata["ys"] == _lbl]
#         _img: th.Tensor = th.reshape(_tdata["xs"][_i], (16, 16))
#         _ax.imshow(th.stack([_img for _ in range(3)], dim=-1))
#         _pxl_locs: th.Tensor = th.argwhere(_tmpl == 1)
#         _ax.scatter(
#             x=_pxl_locs[:, 1],
#             y=_pxl_locs[:, 0],
#             marker="s",
#             color="gold",
#             alpha=0.7,
#             edgecolors="none",
#         )
#     # plot two least likely label
#     if n_leasts == 0:
#         continue
#     _least_label: list[int] = th.argsort(_lbl_dist, descending=False).tolist()[
#         :n_leasts
#     ]
#     for _i, _lbl in enumerate(_least_label):
#         _ax: Axes = _least_axs[_i]
#         _tdata: thd.TensorDict = tdata[tdata["ys"] == _lbl]
#         _img: th.Tensor = th.reshape(_tdata["xs"][_i], (16, 16))
#         _ax.imshow(th.stack([_img for _ in range(3)], dim=-1))
#         _pxl_locs: th.Tensor = th.argwhere(_tmpl == 1)
#         _ax.scatter(
#             x=_pxl_locs[:, 1],
#             y=_pxl_locs[:, 0],
#             marker="s",
#             color="gold",
#             alpha=0.7,
#             edgecolors="none",
#         )
# fig.savefig(
#     os.path.join(out_p, "mnist-tmpl.png"),
#     dpi=720,
#     bbox_inches="tight",
# )
# plt.show()
# plt.close()

# %%
# TODO ident. fives that do not get put into that templates

# %%
# n_tovis: int = len(to_vis_tids)
# _rg = th.Generator().manual_seed(279)
# axs: list[list[Axes]]
# fig, axs = plt.subplots(n_tovis, 8, layout="compressed")
# fig.set_figwidth(1.8 * 8)
# fig.set_figheight(2 * n_tovis)
# for _axs, _tid in zip(axs, to_vis_tids):
#     for _ax in _axs:
#         _ax.set_box_aspect(1.0)
#     # index of instances using current template
#     _idxs: th.Tensor = th.max(tpcomp["rwds"], dim=1).indices == _tid
#     # plot current template
#     _tmpl: th.Tensor = th.reshape(tmpls[_tid], (16, 16))
#     _axs[0].set_title(f"tmpl. {_tid}")
#     _axs[0].imshow(th.stack([_tmpl * 255 for _ in range(3)], dim=-1), cmap="copper")
#     # plot label distribution
#     # class label for instances using ucrrent template
#     _labels: th.Tensor = tdata["ys"][_idxs]
#     # plot label distribution
#     _lbl_dist: th.Tensor = th.bincount(_labels, minlength=n_labels)
#     _axs[1].bar(range(n_labels), _lbl_dist)
#     _axs[1].set_title("cls. dist.")
#     _axs[1].set_ylabel("instances")
#     _axs[1].set_xlabel("cls. label")
#     # plot top 3 labels
#     _top_labels: list[int] = th.argsort(_lbl_dist, descending=True).tolist()[:3]
#     for _i, _lbl in enumerate(_top_labels):
#         _ax: Axes = _axs[_i + 2]
#         _tdata: thd.TensorDict = tdata[tdata["ys"] == _lbl]
#         _img: th.Tensor = th.reshape(_tdata["xs"][_i], (16, 16))
#         _ax.imshow(th.stack([_img for _ in range(3)], dim=-1))
#         _pxl_locs: th.Tensor = th.argwhere(_tmpl == 1)
#         _ax.scatter(
#             x=_pxl_locs[:, 1],
#             y=_pxl_locs[:, 0],
#             marker="s",
#             color="gold",
#             alpha=0.7,
#             edgecolors="none",
#         )
#     # plot two least likely label
#     _least_label: list[int] = th.argsort(_lbl_dist, descending=False).tolist()[:3]
#     for _i, _lbl in enumerate(_least_label):
#         _ax: Axes = _axs[_i + 5]
#         _tdata: thd.TensorDict = tdata[tdata["ys"] == _lbl]
#         _img: th.Tensor = th.reshape(_tdata["xs"][_i], (16, 16))
#         _ax.imshow(th.stack([_img for _ in range(3)], dim=-1))
#         _pxl_locs: th.Tensor = th.argwhere(_tmpl == 1)
#         _ax.scatter(
#             x=_pxl_locs[:, 1],
#             y=_pxl_locs[:, 0],
#             marker="s",
#             color="gold",
#             alpha=0.7,
#             edgecolors="none",
#         )
# fig.tight_layout()
# plt.show()
# plt.close()

# %%
