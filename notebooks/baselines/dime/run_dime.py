# %%
from __future__ import annotations

import mydatasets.aaco
import mymodels.nn
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
from dimelib.cmi_estimator import CMIEstimator
from dimelib.masking_pretrainer import MaskingPretrainModule
from dimelib.utils import MaskLayer
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.plugins.environments as pl_plugins_envs

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("mnist", to_normalize=False)
n_covs: int = tdata["xs"].shape[1]
n_labels: int = len(th.unique(tdata["ys"]))

# %%
tloader = th_data.DataLoader(
    th_data.TensorDataset(tdata["xs"], tdata["ys"]),
    batch_size=128,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=4,
)
vloader = th_data.DataLoader(
    th_data.TensorDataset(vdata["xs"], vdata["ys"]),
    batch_size=128,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)
tstloader = th_data.DataLoader(
    th_data.TensorDataset(tstdata["xs"], tstdata["ys"]),
    batch_size=128,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

# %%
to_share_weights: bool = True

# %%
# make nnet
predictor_nnet = mymodels.nn.make_fcn(
    in_features=n_covs * 2,
    out_features=10,
    layer_specs=[
        (512, th.nn.ReLU, None, 0.3),
        (512, th.nn.ReLU, None, 0.3),
    ],
)
value_nnet = mymodels.nn.make_fcn(
    in_features=n_covs * 2,
    out_features=n_covs,
    layer_specs=[
        (512, th.nn.ReLU, None, 0.3),
        (512, th.nn.ReLU, None, 0.3),
    ],
)
mask_layer = MaskLayer(mask_size=n_covs, append=True)

# %%
if to_share_weights:
    assert isinstance(predictor_nnet, th.nn.Sequential)
    assert isinstance(value_nnet, th.nn.Sequential)
    value_nnet[0] = predictor_nnet[0]
    value_nnet[3] = predictor_nnet[3]

# %%
acc_metric = thm.Accuracy(task="multiclass", num_classes=n_labels)

# %%
masking_pretrain_module = MaskingPretrainModule(
    predictor_nnet,
    mask_layer,
    lr=1e-3,
    loss_fn=th.nn.CrossEntropyLoss(),
    val_loss_fn=acc_metric,
)
masking_trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=200,
    num_sanity_val_steps=0,
    plugins=[pl_plugins_envs.LightningEnvironment()],
)
masking_trainer.fit(masking_pretrain_module, tloader, vloader)

# %%
tfb_logger = pl_loggers.TensorBoardLogger("log")
ckpt_callback = pl_callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor="Perf Val/Final",
    mode="min",
    filename="best_val_perf_model",
    verbose=False,
)

# %%
cmi_estimator_module = CMIEstimator(
    value_nnet,
    predictor_nnet,
    mask_layer,
    lr=1e-3,
    min_lr=1e-6,
    eps_decay=0.2,
    max_features=50,
    eps=0.05,
    loss_fn=th.nn.CrossEntropyLoss(reduction="none"),
    val_loss_fn=acc_metric,
    eps_steps=10,
    patience=5,
)
cmi_trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=200,
    precision=16,
    logger=tfb_logger,
    num_sanity_val_steps=0,
    callbacks=[ckpt_callback],
    plugins=[pl_plugins_envs.LightningEnvironment()],
)
cmi_trainer.fit(cmi_estimator_module, tloader, vloader)

# %%
