from typing import Tuple
import torch
from torch import nn
from touchstone.agents import Agent
from touchstone.buffers import PPOExperience
import numpy as np


class PPOAgent(Agent):
    # TODO implement (note: does not need to fill buffer when playing)
    def play_step(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def explore_step(self, actor_critic: nn.Module, deterministic: bool = True, device: str = 'cpu') -> Tuple[
        float, bool]:
        value, action, action_log_prob = self.get_action(actor_critic, deterministic, device)
        new_state, reward, done, _ = self.env.step(action.detach().cpu().numpy())
        exp = PPOExperience(self.state, action.detach().cpu().numpy(), np.expand_dims(reward, axis=1), done, new_state,
                            action_log_prob.detach().cpu().numpy(), value.detach().cpu().numpy())
        self.buffer.append(exp)
        self.state = new_state

        # TODO investigate if we really need to reset if done when using vectorized environments (i think not)
        # if done:
        #     self.reset()

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

        action_log_prob = action_distribution.log_prob(action)

        return value, action, action_log_prob.unsqueeze(1)

    @staticmethod
    def evaluate_actions(actor_critic: nn.Module, observation_batch, actions_batch):
        values, action_distribution = actor_critic(observation_batch)
        action_log_probs = action_distribution.log_prob(actions_batch)

        return values, action_log_probs
