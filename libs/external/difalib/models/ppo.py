from copy import deepcopy

import torch
import torch.nn as nn

from .model_utils import ConstantLRWithWarmup


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_masks = []
        self.not_dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_masks[:]
        del self.not_dones[:]


class PPO:
    def __init__(self, ac_network, params):
        self.params = vars(params)
        self.lr = self.params.get("policy_lr", 1e-4)
        self.betas = self.params.get("policy_betas", (0.9, 0.999))
        self.eps_clip = self.params.get("eps_clip", 0.2)
        self.k_epochs = self.params.get("k_epochs", 4)
        self.gamma = self.params.get("gamma", 0.99)
        self.ent_reg = self.params.get("ent_reg", 0.001)
        self.batch_size = self.params.get("ppo_batch_size", 1024)
        self.policy = ac_network
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )
        step_size = (
            params.trainset_size
            * self.k_epochs  # number of step is n_features
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

        self.policy_old = deepcopy(ac_network)
        self.mse_loss = nn.MSELoss()
        self.rolling_mean, self.rolling_std = None, None

    def update(self, memory):
        # Setup discounted rewards
        # states  [(B,D), (B1, D), ...
        # not_done [(B,), (B, ), ...
        # reward [(B,), (B1, ),...
        discounted_rewards, mask = [], None
        for reward, not_done in zip(
            reversed(memory.rewards), reversed(memory.not_dones)
        ):
            if discounted_rewards:
                future_reward = self.gamma * discounted_rewards[-1]
                future_mask = mask[not_done]  # subset of current not_dones
                # True means future reward available in future
                reward[future_mask] += future_reward
            discounted_rewards.append(reward)
            mask = not_done
        discounted_rewards.reverse()
        discounted_rewards = torch.cat(discounted_rewards, dim=0)
        rolling_mean = discounted_rewards.mean()
        rolling_std = discounted_rewards.std()
        rewards = (discounted_rewards - rolling_mean) / (rolling_std + 1e-5)

        # convert lists to tensor
        old_states = torch.cat(memory.states, dim=0).detach()
        old_actions = torch.cat(memory.actions, dim=0).detach()
        old_logprobs = torch.cat(memory.logprobs, dim=0).detach()
        old_actionmask = torch.cat(memory.action_masks, dim=0).detach()
        #
        old_dataset = torch.utils.data.TensorDataset(
            old_states,
            old_actions,
            old_logprobs,
            old_actionmask,
            rewards,
        )
        num_element = len(old_states) + 1e-8
        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            old_dataloader = torch.utils.data.DataLoader(
                old_dataset, batch_size=self.batch_size
            )
            self.optimizer.zero_grad()
            for batch in old_dataloader:
                (
                    old_states_,
                    old_actions_,
                    old_logprobs_,
                    old_actionmask_,
                    rewards_,
                ) = batch
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_states_, old_actions_, old_actionmask_
                )

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs_.detach())

                # Finding Surrogate Loss:
                advantages = rewards_ - state_values.detach()
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.mse_loss(state_values, rewards_)
                    - self.ent_reg * dist_entropy
                )

                (loss.sum() / num_element).backward()
            # take gradient step
            if self.params["grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.params["grad_norm"],
                    norm_type="inf",
                )

            self.optimizer.step()
            self.scheduler.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
