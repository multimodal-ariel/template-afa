from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

import _tmplfns
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import mylib.utils
import mymodels.classifiers
import mymodels.nn
import mymodels.protocols
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class MainConf:
    data: Any
    tclassifier: Any
    vclassifier: Optional[Any]
    nnet: Any
    opt: Any
    nnet_tcfg: NeuralNetTrainConf
    make_templates_fn: Any
    init_fidx: int
    lmbda: float
    n_neighs: int
    bsz: int


@dataclass
class NeuralNetTrainConf:
    n_fit_iter: int
    bsz: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


@th.no_grad()
def nnet_cost_est(
    inps: th.Tensor,
    nnet: th.nn.Module,
    lmbda: float,
    tmpls: th.Tensor,
    device: th.device,
) -> th.Tensor:
    n_covs: int = tmpls.shape[1]
    nnet.eval().to(device=device)
    # (n_tmpls, n_covs)
    tmpls = tmpls.to(device=device)
    # (n, n_covs)
    inps = inps.to(device=device)
    fms: th.Tensor = inps[:, n_covs:]
    # (n, n_tmpls)
    cels: th.Tensor = nnet(inps)
    # (n, n_tmpls, n_covs)
    fms_avail: th.Tensor = th.maximum(
        tmpls[None, :, :] - fms[:, None, :], th.as_tensor(0.0, device=device)
    )
    costs: th.Tensor = cels + lmbda * th.sum(fms_avail, dim=2)
    return costs


@th.no_grad()
def compile_selector_dataset(
    tdata: thd.TensorDict, tpcomp: thd.TensorDict
) -> thd.TensorDict:
    # (n_data, n_covs)
    xs: th.Tensor = tdata["xs"]
    # (n_data, )
    ys: th.Tensor = tdata["ys"]
    # (n_data, n_tmpls, n_labels)
    pyhats: th.Tensor = tpcomp["pyhats"]
    # (n_data, n_tmpls)
    cels: th.Tensor = tpcomp["cels"]
    rwds: th.Tensor = tpcomp["rwds"]
    # # (n_data, n_tmpls)
    # slbls: th.Tensor = th.softmax(rwds / tau_rwd, dim=1)
    # bundle tensors into tensordict
    stdata = thd.TensorDict(
        {
            "xs": xs,
            "ys": ys,
            "pyhats": pyhats,
            "cels": cels,
            "rwds": rwds,
        }
    ).auto_batch_size_(1)
    return stdata


@th.no_grad()
def _make_fit_bsinps(
    bstdata: thd.TensorDict, init_fidx: int, tmpls: th.Tensor
) -> th.Tensor:
    bsz: int = len(bstdata)
    n_tmpls: int = len(tmpls)
    # (bsz, )
    btmplidxs: th.Tensor = th.randint(0, n_tmpls, (bsz,))
    # (bsz, n_covs)
    bxs: th.Tensor = bstdata["xs"]
    bfms: th.Tensor = tmpls[btmplidxs].to(device=bxs.device)
    n_covs: int = bxs.shape[1]
    # randomly drop features
    bnms: th.Tensor = th.randint(0, 2, (bsz, n_covs), device=bxs.device)
    bnms = th.clamp(bnms - bfms, 0.0, 1.0)
    bnms[:, init_fidx] = 1
    # (bsz, 2 * n_covs)
    bsinps: th.Tensor = th.cat((bxs * bnms, bnms), dim=1)
    return bsinps


class _TrainState(TypedDict):
    nnet: th.nn.Module
    opt: th.optim.Optimizer
    n_trial_itr: int
    n_fit_itr: int
    opt_step: int


def _fit_iter(
    tstate: _TrainState,
    tloader: th_data.DataLoader,
    init_fidx: int,
    tmpls: th.Tensor,
    pbar: tqdm.tqdm,
    plf: pl.Fabric,
) -> dict[str, float]:
    nnet: th.nn.Module = tstate["nnet"].train().to(device=plf.device)
    opt: th.optim.Optimizer = tstate["opt"]
    slosses_l: list[th.Tensor] = list()
    for bstdata in tloader:
        bstdata: thd.TensorDict
        bstdata = bstdata.to(device=plf.device)
        # (bsz, 2 * n_covs)
        bsinps: th.Tensor = _make_fit_bsinps(
            bstdata=bstdata, init_fidx=init_fidx, tmpls=tmpls
        )
        # (bsz, n_tmpls)
        bsouts: th.Tensor = nnet(bsinps)
        # compute selector loss
        bslosses: th.Tensor = th.nn.functional.mse_loss(
            bsouts, bstdata["cels"], reduction="none"
        )
        bsloss: th.Tensor = th.mean(bslosses)
        # update selector parameter
        opt.zero_grad()
        bsloss.backward()
        opt.step()
        # track metrics
        slosses_l.append(bslosses.detach().to(device="cpu"))
        bmetrics_d: dict[str, float] = {
            "bsloss": bsloss.item(),
        }
        pbar.set_postfix(bmetrics_d)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(bmetrics_d, "train"), step=tstate["opt_step"]
        )
        tstate["opt_step"] = tstate["opt_step"] + 1
    sloss_avg: th.Tensor = th.mean(th.cat(slosses_l, dim=0))
    metrics_d: dict[str, float] = {"sloss_avg": sloss_avg.item()}
    return metrics_d


def fit(
    tstate: _TrainState,
    stdata: thd.TensorDict,
    init_fidx: int,
    tmpls: th.Tensor,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
):
    tloader = th_data.DataLoader(
        stdata,  # type: ignore
        batch_size=bsz,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    pbar = tqdm.trange(n_iter, dynamic_ncols=True, leave=True)
    for _ in pbar:
        metrics_d: dict[str, float] = _fit_iter(
            tstate=tstate,
            tloader=tloader,
            init_fidx=init_fidx,
            tmpls=tmpls,
            pbar=pbar,
            plf=plf,
        )
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(metrics_d, "train"), step=tstate["n_fit_itr"]
        )
        tstate["n_fit_itr"] = tstate["n_fit_itr"] + 1
    pbar.close()


@hd.main(version_base=None)
def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    _tdata: thd.TensorDict
    vdata: thd.TensorDict
    tstdata: thd.TensorDict
    _tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = _tdata["xs"].shape[1]
    n_labels: int = len(th.unique(_tdata["ys"]))
    # split training data into two for classifier and afa
    _tdata_shuffle_idxs = th.randperm(
        len(_tdata), generator=th.Generator().manual_seed(279)
    )
    tdata: thd.TensorDict
    extdata: thd.TensorDict
    tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
    extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
    # make classifier
    tclassifier: mymodels.classifiers.SubsetFeatureClassifier = hd.utils.instantiate(
        cfg.tclassifier, xs_train=extdata["xs"].numpy(), ys_train=extdata["ys"].numpy()
    )
    vclassifier: mymodels.classifiers.SubsetFeatureClassifier = tclassifier
    if cfg.vclassifier is not None:
        vclassifier = hd.utils.instantiate(
            cfg.vclassifier,
            xs_train=extdata["xs"].numpy(),
            ys_train=extdata["ys"].numpy(),
        )
    # configure logger and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")  # type: ignore
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")  # type: ignore
    plf_tmpl = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="cpu")
    plf_nnet = pl.Fabric(loggers=[tfb_logger, csv_logger], accelerator="auto")
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    # make templates
    make_templates_fn: Callable = hd.utils.instantiate(
        cfg.make_templates_fn, _partial_=True
    )
    tmpls: th.Tensor = make_templates_fn(
        tdata=tdata,
        classifier=tclassifier,
        init_fidx=cfg.init_fidx,
        lmbda=cfg.lmbda,
        bsz=cfg.bsz,
        plf=plf_tmpl,
    )
    th.save(_tdata_shuffle_idxs, os.path.join(output_dir, "tdata_shuffle_idxs.pt"))
    th.save(tmpls, os.path.join(output_dir, "tmpls.pt"))
    # make nnet
    nnet: th.nn.Module = hd.utils.instantiate(
        cfg.nnet, in_features=2 * tdata["xs"].shape[1], out_features=len(tmpls)
    )
    opt: th.optim.Optimizer = hd.utils.instantiate(cfg.opt, params=nnet.parameters())
    # fit nnet cost estimator
    tpcomp: thd.TensorDict = _tmplfns.precomp_rwds_for_tmpls(
        tmpls=tmpls, data=tdata, classifier=tclassifier, lmbda=cfg.lmbda, bsz=cfg.bsz
    )
    tstate = fit(
        tstate=_TrainState(nnet=nnet, opt=opt, n_trial_itr=0, n_fit_itr=0, opt_step=0),
        stdata=compile_selector_dataset(tdata, tpcomp),
        init_fidx=cfg.init_fidx,
        tmpls=tmpls,
        n_iter=cfg.nnet_tcfg.n_fit_iter,
        bsz=cfg.nnet_tcfg.bsz,
        plf=plf_nnet,
    )
    plf_nnet.save(os.path.join(output_dir, "tstate.pt"), tstate)
    # evaluate validation set performance with knn
    metrics_d: dict[str, float] = _tmplfns.evaluate(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: _tmplfns.knn_cost_est(
            x,
            lmbda=cfg.lmbda,
            txs=tdata["xs"],
            tcels=tpcomp["cels"],
            tmpls=tmpls,
            n_neighs=cfg.n_neighs,
            p=2,
        ),
        init_fidx=cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf_tmpl,
    )
    plf_tmpl.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_knn"))
    # evaluate validation set performance with nnet
    metrics_d = _tmplfns.evaluate(
        data=vdata,
        classifier=vclassifier,
        cost_est=lambda x: nnet_cost_est(
            x, nnet=nnet, lmbda=cfg.lmbda, tmpls=tmpls, device=plf_nnet.device
        ),
        init_fidx=cfg.init_fidx,
        tmpls=tmpls,
        metrics_func=metrics_func,
        plf=plf_tmpl,
    )
    plf_tmpl.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "eval_nnet"))
    # logger flush record and close
    tfb_logger.finalize("success")
    csv_logger.finalize("success")


if __name__ == "__main__":
    main()
