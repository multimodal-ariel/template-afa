from copy import deepcopy
import os

import torch

from .data_utils import load_data, prepare_data
from .model_utils import (
    assign_optimizer_scheduler,
    compute_batch_loss,
    compute_loss,
    compute_metric,
)
from .network_utils import get_imputation_model


class Model(object):
    def __init__(self, params, device):
        self.params = deepcopy(params)
        self.device = device
        self.data_parameters = prepare_data(self)
        self.pred_model = get_imputation_model(self.params, self.data_parameters).to(
            self.device
        )
        assign_optimizer_scheduler(self, self.params.iters)

    def save(self):
        model_dir: str = os.path.join(self.params.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.pred_model.state_dict(),
                # "params": self.params,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "data_parameters": self.data_parameters,
            },
            os.path.join(model_dir, f"{self.params.name}.pt"),
        )

    def load(self, checkpoint):
        self.pred_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def train_step(self, batch):
        self.pred_model.train()
        vlb = self.pred_model.batch_vlb(batch["batch"], batch["mask"]).mean()
        (-vlb).backward()
        if self.params.grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.pred_model.parameters(),
                max_norm=self.params.grad_norm,
                norm_type="inf",
            )
        self.optimizer.step()
        self.scheduler.step()
        return {"vlb": float(vlb.detach().cpu().numpy()), "size": len(batch["batch"])}

    def test_step(self, batch):
        self.pred_model.eval()
        with torch.no_grad():
            vlb = self.pred_model.batch_iwae(batch["batch"], batch["mask"], K=5).mean()
            return {
                "vlb": float(vlb.detach().cpu().numpy()),
                "size": len(batch["batch"]),
            }

    def dataloaders(self, iters=None):
        return load_data(self)
