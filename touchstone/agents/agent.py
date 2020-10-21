from typing import Any
import gym
import torch

from touchstone.buffers import Buffer


class Agent:
    def __init__(self, env: gym.Env, buffer: Buffer = None) -> None:
        self.env = env
        self.buffer = buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def play_step(self, *args, **kwargs):
        raise NotImplementedError
