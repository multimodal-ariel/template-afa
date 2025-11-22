from __future__ import annotations

import logging
import math
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import mymodels
import tafalib
import tensordict as thd
import torch as th
import tqdm.auto as tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch.utils.data as th_data


@dataclass
class MainConf:
    data: Any
    n_masks: int
    nnet: Any
    nnet_fit_cfg: NeuralNetFitConf
    plf: Any


@dataclass
class NeuralNetFitConf:
    opt_type: Any
    opt_kwargs: dict[str, Any]
    n_iter: int
    bsz: int
    eval_every_n_iter: int


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


def make_feature_masks(
    n_covs: int,
    n_masks: int,
    min_features: int,
    max_features: Optional[int],
) -> th.Tensor:
    max_features = n_covs if max_features is None else max_features
    bincnt_fcs_l: list[int] = [
        # in order to accomondate for init_fidx,
        # both n_covs and i is one less than desired n_feats
        min(math.comb(n_covs, i), th.iinfo(th.long).max)
        for i in range(min_features, max_features + 1)
    ]
    n_masks = min(n_masks, sum(bincnt_fcs_l))
    bincnt_fcs: th.Tensor = th.as_tensor(bincnt_fcs_l, dtype=th.long)
    ps: th.Tensor = th.ones_like(bincnt_fcs, dtype=th.float32)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_masks, replacement=True), minlength=len(bincnt_fcs)
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(_curr_bincnts >= bincnt_fcs, 0.0, 1.0)
        _realloc_cnts: th.Tensor = th.where(
            _curr_bincnts > bincnt_fcs, _curr_bincnts - bincnt_fcs, 0
        )
        _tmp_bincnts: th.Tensor = th.bincount(
            th.multinomial(
                _tmp_ps,
                int(th.sum(_realloc_cnts).item()),
                replacement=True,
            ),
            minlength=len(bincnt_fcs),
        )
        _curr_bincnts = _curr_bincnts - _realloc_cnts + _tmp_bincnts
    nfc_from_each_binned_fcs = _curr_bincnts
    # make unique feature combination
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [set() for _ in bincnt_fcs]
    for _k, (_count, _fcs_set) in enumerate(
        zip(nfc_from_each_binned_fcs, fcs_sets_by_bins)
    ):
        if _count == 0:
            continue
        _nfeats: int = _k + min_features
        while len(_fcs_set) < _count:
            _fc_l: list[int] = th.multinomial(
                th.ones((n_covs,)), num_samples=_nfeats
            ).tolist()
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc = tuple(_fc_l)
            if _fc not in _fcs_set:
                _fcs_set.add(_fc)
    # from fcomb to act
    fms: th.Tensor = th.zeros((n_masks, n_covs), dtype=th.long)
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    assert len(fms) == len(fcs_l)
    for _i, _fc in enumerate(fcs_l):
        fms[_i, _fc] = 1
    return fms


@th.enable_grad()
def fit_nnet_clf(
    nnet: mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier,
    tdata: thd.TensorDict,
    n_masks: int,
    opt: th.optim.Optimizer,
    n_iter: int,
    bsz: int,
    plf: pl.Fabric,
    vdata: Optional[thd.TensorDict],
    eval_every_n_iter: int,
    tstdata: Optional[thd.TensorDict],
):
    n_covs: int = tdata["xs"].shape[1]
    # fit classifier
    pbar = tqdm.trange(
        n_iter,
        desc="fit_subset_cls",
        leave=False,
        dynamic_ncols=True,
    )
    tloader = th_data.DataLoader(
        tdata, batch_size=bsz, shuffle=True, collate_fn=lambda x: x
    )
    nnet = nnet.to(device=plf.device)
    _step: int = 0
    for _itr in pbar:
        nnet.train()
        _bpbar = tqdm.tqdm(
            tloader,
            total=math.ceil(len(tdata) / bsz),
            desc="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for _btdata in _bpbar:
            if len(_btdata) == 1:
                continue
            _btdata: thd.TensorDict = _btdata.to(device=plf.device)
            _bsz: int = len(_btdata)
            _bfms: th.Tensor = tafalib.makers.candidates.make_feature_masks(
                n_covs=n_covs,
                n_masks=n_masks,
                min_features=1,
                max_features=None,
                generator=None,
            )
            _nms: int = len(_bfms)
            _bfms = _bfms[None, :, :].expand(_bsz, -1, -1).to(device=plf.device)
            _bxs: th.Tensor = _btdata["xs"][:, None, :].expand(-1, _nms, -1)
            _binps: th.Tensor = th.cat((_bxs * _bfms, _bfms), dim=2).flatten(0, 1)
            _bys: th.Tensor = _btdata["ys"][:, None].expand(-1, _nms).flatten(0, 1)
            _bouts: th.Tensor = nnet.nnet(_binps)
            _bcel: th.Tensor = th.nn.functional.cross_entropy(_bouts, _bys)
            opt.zero_grad()
            _bcel.backward()
            opt.step()
            _bpbar.set_postfix({"bcel": _bcel.item()})
            plf.log_dict({"train_rolling/bcel": _bcel.item()}, step=_step)
            _step = _step + 1
        _bpbar.close()
        if _itr % eval_every_n_iter == 0:
            _tmpls: th.Tensor = tafalib.makers.candidates.make_feature_masks(
                n_covs=n_covs,
                n_masks=n_masks,
                min_features=1,
                max_features=None,
                generator=None,
            ).to(device=plf.device)
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(
                    nnet.evaluate(tdata, tmpls=_tmpls, init_fidx=None),
                    "train_rolling",
                ),
                step=_itr,
            )
            if vdata is not None:
                plf.log_dict(
                    mylib.utils.add_prefix_to_dict(
                        nnet.evaluate(vdata, tmpls=_tmpls, init_fidx=None),
                        "validation_rolling",
                    ),
                    step=_itr,
                )
    pbar.close()
    # eval classifier
    if tstdata is not None:
        _tmpls: th.Tensor = tafalib.makers.candidates.make_feature_masks(
            n_covs=n_covs,
            n_masks=n_masks,
            min_features=1,
            max_features=None,
            generator=None,
        ).to(device=plf.device)
        plf.log_dict(
            mylib.utils.add_prefix_to_dict(
                nnet.evaluate(tstdata, tmpls=_tmpls, init_fidx=None),
                "test_eot",
            ),
            step=n_iter,
        )
    return


def main(cfg: MainConf):
    # _delay_import()
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make dataset
    tdata, vdata, tstdata = hd.utils.call(cfg.data)
    n_covs: int = tdata["xs"].shape[1]
    n_labels: int = len(th.unique(tdata["ys"]))
    # make nnet
    nnet: th.nn.Module = hd.utils.instantiate(
        cfg.nnet,
        in_features=n_covs * 2,
        out_features=n_labels,
    )
    classifier = mymodels.classifiers.SubsetFeatureConcatNeuralNetClassifier(
        nnet=nnet,
        xs_train=tdata["xs"],
        ys_train=tdata["ys"],
        fit_kwargs=hd.utils.instantiate(cfg.nnet_fit_cfg),
    )

    # fit classifier
    opt = cfg.nnet_fit_cfg.opt_type(
        classifier.parameters(), **cfg.nnet_fit_cfg.opt_kwargs
    )
    tfb_logger = plf_loggers.TensorBoardLogger(output_dir, name="", version="")
    csv_logger = plf_loggers.CSVLogger(tfb_logger.log_dir, name="", version="")
    plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger],
        plugins=[plf_plugins_envs.LightningEnvironment()],
    )
    fit_nnet_clf(
        nnet=classifier,
        tdata=tdata,
        n_masks=cfg.n_masks,
        opt=opt,
        n_iter=cfg.nnet_fit_cfg.n_iter,
        bsz=cfg.nnet_fit_cfg.bsz,
        plf=plf,
        vdata=vdata,
        eval_every_n_iter=cfg.nnet_fit_cfg.eval_every_n_iter,
        tstdata=tstdata,
    )
    # save classifier
    th.save(classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
    # log metrics
    # make feature masks
    tmpls: th.Tensor = make_feature_masks(
        n_covs=n_covs, n_masks=max(cfg.n_masks, 2048), min_features=1, max_features=None
    )  # configure loggers
    vmetrics_d: dict[str, float] = classifier.evaluate(vdata, tmpls, init_fidx=None)
    plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "val"))
    # cleanup
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
