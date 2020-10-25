import torch
from touchstone.policies import PPOPolicy


def make_continuous_random_policy(action_space, device):
    def policy_fn(*agrs, **kwargs):
        if device is not None:
            return torch.from_numpy(action_space.sample())
        else:
            return action_space.sample()

    return policy_fn


def make_ppo_policy(observation_shape, action_shape):
    return PPOPolicy(observation_shape, action_shape)
