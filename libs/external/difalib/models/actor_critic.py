import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index.squeeze(1)


def hard_gumbel_sample(logits, dim=-1, tau=1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = F.softmax(gumbels, dim=-1)
    # y_soft = gumbels.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index.squeeze(1)


class Actor(nn.Module):
    def __init__(self, shared_layer, actor_layer, tau=1):
        super().__init__()
        # actor
        self.tau = tau
        self.shared_layer = shared_layer
        self.actor_layer = actor_layer

    def forward(self, state, action_mask):
        hidden_state = self.shared_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(
            torch.log(action_mask.float()), min=torch.finfo(torch.float32).min
        )
        logits = logits + inf_mask
        probs = F.softmax(logits, dim=-1)  # N, D
        # dist = Categorical(probs)
        # dist_entropy = dist.entropy()
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-20), dim=-1))

        train_mask, actions = hard_gumbel_sample(logits, tau=self.tau)
        return train_mask, actions, entropy


class ActorCritic(nn.Module):
    def __init__(self, shared_layer, actor_layer, value_layer):
        super().__init__()
        self.shared_layer = shared_layer
        self.actor_layer = actor_layer
        self.value_layer = value_layer

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, action_mask):
        hidden_state = self.shared_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(
            torch.log(action_mask.float()), min=torch.finfo(torch.float32).min
        )
        logits = logits + inf_mask
        softmax = nn.Softmax(dim=-1)
        action_probs = softmax(logits)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.action_masks.append(action_mask)
        memory.logprobs.append(dist.log_prob(action))

        return action.detach()

    def evaluate(self, state, action, action_mask):
        hidden_state = self.shared_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(
            torch.log(action_mask.float()), min=torch.finfo(torch.float32).min
        )
        logits = logits + inf_mask
        softmax = nn.Softmax(dim=-1)
        action_probs = softmax(logits)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(hidden_state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
