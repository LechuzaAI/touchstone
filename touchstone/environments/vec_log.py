import numpy as np
from touchstone.environments import VecEnvWrapper


class VecLog(VecEnvWrapper):
    def __init__(self, venv):
        super(VecLog, self).__init__(venv)
        self.returns = []
        self.mean_reward_per_step = 0
        self.total_steps = 0
        self.episode_rewards = []

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        self.returns.append(reward)
        self.total_steps += 1
        if all(done):
            self.episode_rewards.append(sum(self.returns))
            self.returns = []
            self.mean_reward_per_step = sum(self.episode_rewards) / self.total_steps
        return obs, reward, done, info

    def reset(self):
        state = self.venv.reset()
        self.returns = []
        self.mean_reward_per_step = 0
        self.total_steps = 0
        return state
