import torch


class PPOExperienceBuffer(object):
    def __init__(self, T, N, observation_shape, action_space_shape):
        super().__init__()
        self.T = T
        self.N = N
        self.observation_space = observation_shape
        self.action_space_shape = action_space_shape
        self.observations = torch.zeros(T + 1, N, *observation_shape)
        self.rewards = torch.zeros(T, N, 1)
        self.values = torch.zeros(T, N, 1)
        self.returns = torch.zeros(T, N, 1)
        self.actions_log_probs = torch.zeros(T, N, 1)
        self.actions = torch.zeros(T, N, *action_space_shape)
        self.step = 0
        self.device = "cpu"

    def to(self, *args, **kwargs):
        self.observations = self.observations.to(*args, **kwargs)
        self.rewards = self.rewards.to(*args, **kwargs)
        self.values = self.values.to(*args, **kwargs)
        self.returns = self.rewards.to(*args, **kwargs)
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
