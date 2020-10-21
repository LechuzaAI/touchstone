from typing import Any

import torch.nn as nn

from touchstone.nets import MLPNet


class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_shape, base=MLPNet):
        super(ActorCriticNet, self).__init__()
        self.actor = base(observation_shape[0], action_shape[0], dist_output=True)
        self.critic = base(observation_shape[0], 1)

    def forward(self, x):
        action_distribution = self.actor(x)
        value = self.critic(x)
        return value, action_distribution
