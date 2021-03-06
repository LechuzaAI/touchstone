import torch.nn as nn

from touchstone.nets import MLPNet


class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_shape, base=MLPNet, action_std=1.0):
        super(ActorCriticNet, self).__init__()
        self.actor = base(observation_shape[0], action_shape[0], dist_output=True, dist_std=action_std)
        self.critic = base(observation_shape[0], 1)

    def forward(self, x):
        action_distribution = self.actor(x)
        value = self.critic(x)
        return value, action_distribution

    def to(self, *args, **kwargs):
        super(ActorCriticNet, self).to(*args, **kwargs)
        self.actor.to(*args, **kwargs)
        self.critic.to(*args, **kwargs)
