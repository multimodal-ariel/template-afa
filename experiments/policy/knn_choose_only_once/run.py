from __future__ import annotations

import itertools as itrtls
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib
import mymodels
import tafalib
import tensordict as thd
import torch as th
import torchmetrics as thm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    mktmpl_exp: Optional[MakeTemplateExpConf]
    mktmpl_run: Optional[str]
    plf: Any


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


@th.no_grad()
def run_one_episode_select_only_once_all_obsd(
    x: th.Tensor,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    plf: pl.Fabric,
) -> tuple[th.Tensor, list[int], tuple[int, ...]]:
    """select once and run one episode using **all** features acquired

    Args:
        x (th.Tensor): (n_covs, ) instance of interest
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): a cost estimator mapping a tensor of shape (n, 2 * n_covs) to a tensor of (n, n_tmpls)
        init_fidx (int): initial feature indices
        tmpls (th.Tensor): (n_tmpls, n_covs)
        plf (pl.Fabric): lightning fabric instance

    Returns::
        th.Tensor: (n_labels, ) predicted class probability
        list[int]: (n_feats_obsd, ) a list of feature indices in the order of acquisition.
        tuple[int, ...]]: (n_feats_obsd, ) a tuple of feature indices acquired; does NOT preseve acquisition order
    """
    classifier.eval().to(device=plf.device)
    if isinstance(cost_est, th.nn.Module):
        cost_est.eval().to(device=plf.device)
    x = x.to(device=plf.device)
    tmpls = tmpls.to(device=plf.device)
    fobsd_l: list[int] = [init_fidx]
    tmpl_idx: int | None = None
    for _i in itrtls.count():
        # make feature bit mask
        _fm: th.Tensor = th.zeros_like(x)
        _fm[fobsd_l] = 1
        # forward prop. cost est.
        # (1, 2 * n_covs)
        _inps: th.Tensor = th.cat((x * _fm, _fm))[None, :]
        # (1, n_tmpls)
        if _i == 0:
            _costs: th.Tensor = cost_est(_inps)
            tmpl_idx = int(th.argmin(_costs[0]).item())
        assert tmpl_idx is not None
        _fm_avail: th.Tensor = th.maximum(
            tmpls[tmpl_idx] - _fm, th.as_tensor(0.0, device=plf.device)
        )
        if th.sum(_fm_avail) == 0:
            break
        fobsd_l.append(int(th.argmax(_fm_avail).item()))
    fcomb = tuple(sorted(fobsd_l))
    acts: th.Tensor = th.zeros((1, x.shape[0]), dtype=th.long, device=x.device)
    acts[0, fobsd_l] = 1
    pyhats: th.Tensor = classifier.predict_proba(x[None, :] * acts, acts)
    return pyhats[0], fobsd_l, fcomb


def evaluate_select_only_once(
    data: thd.TensorDict,
    classifier: mymodels.classifiers.SubsetFeatureClassifier,
    cost_est: Callable[[th.Tensor], th.Tensor],
    init_fidx: int,
    tmpls: th.Tensor,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
) -> dict[str, float]:
    """compute metrics through rolling out policy over a dataset

    Args:
        data (thd.TensorDict): (n, ) dataset of interest; must contain `xs` and `ys`
        classifier (mymodels.classifiers.SubsetFeatureClassifier): subset feature classifier used to make prediction
        cost_est (Callable[[th.Tensor], th.Tensor]): cost estimator
        init_fidx (int): initial feature index
        tmpls (th.Tensor): (n_tmpls, n_covs) collection of templates
        metrics_func (thm.MetricCollection): a collection of metrics of interests
        plf (pl.Fabric): plf instance for prediction

    Returns:
        dict[str, float]: a dictionary of metrics
    """
    snfobsd_l: list[int] = list()
    snfcomb_l: list[int] = list()
    metrics_func.reset()
    for _data in data:
        _pyhat, _fobsd_l, _fcomb = run_one_episode_select_only_once_all_obsd(
            x=_data["xs"],
            classifier=classifier,
            cost_est=cost_est,
            init_fidx=init_fidx,
            tmpls=tmpls,
            plf=plf,
        )
        snfobsd_l.append(len(_fobsd_l))
        snfcomb_l.append(len(_fcomb))
        metrics_func.update(
            _pyhat[None, :].to(device="cpu"), _data["ys"][None].to(device="cpu")
        )
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    metrics_d.update(
        {
            "init_fidx": init_fidx,
            "feature observed": th.mean(
                th.as_tensor(snfobsd_l, dtype=th.float32)
            ).item(),
            "feature used": th.mean(th.as_tensor(snfcomb_l, dtype=th.float32)).item(),
        }
    )
    return metrics_d


def _get_mktmpl_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "mktmpl_run") and cfg.mktmpl_run is not None:
        return cfg.mktmpl_run
    assert hasattr(cfg, "mktmpl_exp") and cfg.mktmpl_exp is not None
    return os.path.join(cfg.mktmpl_exp.exp_p, str(cfg.mktmpl_exp.run_id))


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # get experiment dir
    mktmpl_run_dir: str = _get_mktmpl_run_dir(cfg)
    # load saved result from make_templates
    tmpls: th.Tensor = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tmpls.pt"),
        weights_only=False,
    )
    tpcomp: thd.TensorDict = th.load(
        os.path.join(mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tpcomp.pt"),
        weights_only=False,
    )
    mktmpl_cfg = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), mktmpl_run_dir, ".hydra", "config.yaml"
        )
    )
    OmegaConf.save(mktmpl_cfg, os.path.join(output_dir, ".hydra", "mktmpl_config.yaml"))
    # make dataset
    _tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    _tdata, vdata, tstdata = hd.utils.call(mktmpl_cfg.data)
    n_covs: int = _tdata["xs"].shape[1]
    n_labels: int = len(th.unique(_tdata["ys"]))
    # split training data into two for classifier and afa
    _tdata_shuffle_idxs: th.Tensor = th.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tdata_shuffle_idxs.pt"
        )
    )
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
    extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
    # make classifier
    tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
        mktmpl_cfg.tclassifier,
        xs_train=extdata["xs"].numpy(),
        ys_train=extdata["ys"].numpy(),
    )
    if isinstance(tclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
        tclassifier_p = os.path.join(
            mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
        )
        if os.path.exists(tclassifier_p):
            tclassifier.load_state_dict(th.load(tclassifier_p, map_location="cpu"))
        else:
            tclassifier.fit_(tmpls)
    vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
    if mktmpl_cfg.vclassifier is not None:
        vclassifier = hd.utils.instantiate(
            mktmpl_cfg.vclassifier,
            xs_train=extdata["xs"].numpy(),
            ys_train=extdata["ys"].numpy(),
        )
        if isinstance(vclassifier, mymodels.classifiers.SubsetFeatureConcatClassifier):
            vclassifier_p = os.path.join(
                mylib.utils.get_project_root_dir(), mktmpl_run_dir, "tclassifier.pt"
            )
            if os.path.exists(vclassifier_p):
                vclassifier.load_state_dict(th.load(vclassifier_p, map_location="cpu"))
            else:
                vclassifier.fit_(tmpls)
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
    plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger]
    )
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    # evaluate validation set performance with select only once nearest neighbors
    metrics_d = evaluate_select_only_once(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: tafalib.functional.knn_cost_est(
            x,
            lmbda=mktmpl_cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=mktmpl_cfg.n_neighs,
            p=2,
        ),
        init_fidx=mktmpl_cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf,
    )
    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval"))
    # logger flush record and close
    tfb_logger.finalize("success")
    csv_logger.finalize("success")


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
