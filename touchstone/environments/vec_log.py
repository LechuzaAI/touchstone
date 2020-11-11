import numpy as np
from touchstone.environments import VecEnvWrapper


class VecLog(VecEnvWrapper):
    def __init__(self, venv):
        super(VecLog, self).__init__(venv)
        self.returns = []
        self.mean_reward_per_step = 0
        self.total_steps = 0

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        self.returns.append(reward)
        self.total_steps += 1
        self.mean_reward_per_step = np.array(self.returns).mean() / self.total_steps
        if all(done):
            self.returns = []
            self.mean_reward_per_step = 0
            self.total_steps = 0
        return obs, reward, done, info

    def reset(self):
        state = self.venv.reset()
        self.returns = []
        self.mean_reward_per_step = 0
        self.total_steps = 0
        return state
