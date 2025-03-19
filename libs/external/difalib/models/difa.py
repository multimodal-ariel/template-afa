import torch

from .model_utils import ConstantLRWithWarmup


class DIFA:
    def __init__(self, actor_network, params):
        self.params = vars(params)

        self.lr = max(self.params["base_lr"], self.params.get("policy_lr", 1e-4))
        self.betas = self.params.get("policy_betas", (0.9, 0.999))
        self.gamma = self.params.get("gamma", 0.99)

        self.policy = actor_network
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )
        step_size = (
            params.trainset_size
            * params.n_features  # number of step is n_features
            * params.iters
        ) / (
            params.batch_size * params.cycle
        )  # model batch size, not ppo batch size

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=params.policy_base_lr,
            max_lr=self.lr,
            cycle_momentum=False,
            mode="triangular",
            step_size_up=step_size,
        )

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        if self.params["grad_norm"] > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=self.params["grad_norm"],
                norm_type="inf",
            )

        self.optimizer.step()
