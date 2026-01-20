import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np


class SurrogateClassifier(nn.Module):
    """
    Differentiable classifier
    """

    def __init__(self, input_dim, n_classes, hidden_dim=256, dataset_name=""):
        super().__init__()
        # x_o + mask
        if dataset_name == "cube":
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, n_classes),
            )
        elif dataset_name == "fashionfull":
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
        elif dataset_name == "engine":
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )

    def forward(self, x, mask):
        masked_input = torch.cat([x * mask, mask], dim=-1)
        return self.net(masked_input)


class TemplatePolicy(nn.Module):
    def __init__(
        self,
        templates,
        surrogate_clf,
        input_dim,
        start_dim,
        hidden_dim=256,
        temp=2.0,
        optimize_templates=False,
        dataset_name="",
    ):
        super().__init__()
        self.surrogate = surrogate_clf
        self.M, self.d = templates.shape
        self.temp = temp
        self.optimize_templates = optimize_templates
        self.start_dim = start_dim

        self.register_buffer("start_mask", torch.zeros(1, self.d))
        self.start_mask[0, start_dim] = 1.0

        self.register_buffer("optimize_mask", torch.ones(1, self.d))
        self.optimize_mask[0, start_dim] = 0.0  # no optimzation for first feat

        # prune mask
        self.register_buffer("keep_mask", torch.ones(self.M))

        # Policy Network
        if dataset_name == "cube":
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, self.M),
            )
        elif dataset_name == "fashionfull":
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.M),
            )
        elif dataset_name == "engine":
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, self.M),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.M),
            )

        if self.optimize_templates:
            # soft init
            init_logits = torch.zeros_like(templates)
            init_logits[templates > 0.5] = 3.0
            init_logits[templates <= 0.5] = -3.0
            self.template_logits = nn.Parameter(init_logits)
        else:
            self.register_buffer("fixed_templates", templates.detach())

    @property
    def templates(self):
        if self.optimize_templates:
            M = torch.sigmoid(self.template_logits)
            final_templates = M * self.optimize_mask + self.start_mask
            return final_templates

        return self.fixed_templates

    def get_hard_templates(self):
        # helper for eval
        if self.optimize_templates:
            M = (self.template_logits > 0).float()
            return M * self.optimize_mask + self.start_mask
        return self.fixed_templates

    def forward(self, x, mask):
        inp = torch.cat([x * mask, mask], dim=-1)
        logits = self.net(inp)
        if hasattr(self, "keep_mask"):
            # dont use pruned templates
            logits = logits + (1.0 - self.keep_mask) * -1e9
        return logits

    def loss(self, x, y, mask, alpha):
        logits = self.forward(x, mask)

        choice_proxy = F.gumbel_softmax(logits, tau=self.temp, hard=True, dim=-1)

        soft_templates = self.templates
        target_soft = torch.matmul(choice_proxy, soft_templates)

        target_hard = (target_soft > 0.5).float()
        target_proxy = target_soft + (target_hard - target_soft).detach()

        next_mask_proxy = torch.maximum(mask, target_proxy)

        pred_logits = self.surrogate(x, next_mask_proxy)
        pred_loss = F.cross_entropy(pred_logits, y.view(-1), reduction="none")

        needed = torch.relu(target_proxy - mask)
        acq_cost = alpha * torch.sum(needed, dim=-1)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        total_loss = pred_loss + acq_cost - (0.001 * entropy)

        return total_loss.mean()
