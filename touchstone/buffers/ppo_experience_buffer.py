import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler


class PPOExperienceBuffer(object):
    def __init__(self, T, N, M, observation_shape, action_space_shape):
        super().__init__()
        self.T = T
        self.N = N
        self.M = M
        self.observation_space = observation_shape
        self.action_space_shape = action_space_shape
        self.observations = torch.zeros(T + 1, N, *observation_shape)
        self.rewards = torch.zeros(T, N, 1)
        self.values = torch.zeros(T, N, 1)
        self.advantage_estimates = torch.zeros(T, N, 1)
        self.actions_log_probs = torch.zeros(T, N, 1)
        self.actions = torch.zeros(T, N, *action_space_shape)
        self.step = 0
        self.device = "cpu"
        self.iter_index = 0

    def to(self, *args, **kwargs):
        self.observations = self.observations.to(*args, **kwargs)
        self.rewards = self.rewards.to(*args, **kwargs)
        self.values = self.values.to(*args, **kwargs)
        self.advantage_estimates = self.rewards.to(*args, **kwargs)
        self.actions_log_probs = self.actions_log_probs.to(*args, **kwargs)
        self.actions = self.actions.to(*args, **kwargs)
        self.device = args[0]

    def insert(self, observation, action, action_log_probs, value, reward):
        self.observations[self.step + 1].copy_(observation)
        self.actions[self.step].copy_(action)
        self.actions_log_probs[self.step].copy_(action_log_probs)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.step = (self.step + 1) % self.T

    def push(self, observation):
        self.observations[self.step].copy_(observation)

    def __getitem__(self, t):
        return self.observations[t]

    def __iter__(self):
        buffer_size = self.T * self.N
        minibatch_size = self.M
        sampler = BatchSampler(SubsetRandomSampler(range(buffer_size)), minibatch_size, drop_last=True)
        for i in sampler:
            try:
                obs_batch = self.observations[:-1].view(-1, *self.observations.size()[2:])[i]
                actions_batch = self.actions.view(-1, self.actions.size(-1))[i]
                values_batch = self.values.view(-1, 1)[i]
                advantages_batch = self.advantage_estimates.view(-1, 1)[i]
                old_action_log_probs_batch = self.actions_log_probs.view(-1, 1)[i]
                yield obs_batch, actions_batch, values_batch, advantages_batch, old_action_log_probs_batch
            except IndexError:
                pass