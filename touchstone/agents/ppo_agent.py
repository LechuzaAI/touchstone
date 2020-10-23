from typing import Tuple

import torch
from torch import nn

from touchstone.agents import Agent
from touchstone.buffers import Experience, PPOExperience


class PPOAgent(Agent):
    @torch.no_grad()
    def play_step(self, actor_critic: nn.Module, deterministic: bool = True, device: str = 'cpu') -> Tuple[float, bool]:
        value, action, action_log_prob = self.get_action(actor_critic, deterministic)
        new_state, reward, done, _ = self.env.step(action)
        exp = PPOExperience(self.state, action, reward, done, new_state, action_log_prob, value)
        self.buffer.append(exp)
        self.state = new_state

        if done:
            self.reset()

        return reward, done

    def get_action(self, actor_critic: nn.Module, deterministic: bool = False, device: str = 'cpu'):
        if not isinstance(self.state, torch.Tensor):
            state = torch.tensor(self.state, device=device)
        else:
            state = self.state
        value, action_distribution = actor_critic(state)

        if deterministic:
            action = action_distribution.mode
        else:
            action = action_distribution.sample()

        action_log_prob = action_distribution.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_prob

    def evaluate_actions(self, actor_critic: nn.Module, observation_batch, actions_batch):
        values, action_distribution = actor_critic(observation_batch)
        action_log_probs = action_distribution.log_prob(actions_batch).sum(-1, keepdim=True)

        return values, action_log_probs
