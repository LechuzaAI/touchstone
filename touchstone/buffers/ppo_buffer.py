from dataclasses import dataclass, astuple
import numpy as np
from touchstone.buffers import PPOExperience, Buffer


class PPOBuffer(Buffer):
    def __init__(self, size: int):
        super(PPOBuffer, self).__init__(size)
        self.advantages = None

    def append(self, experience: PPOExperience) -> None:
        self.buffer.append(experience)

    def update_advantages(self, advantages):
        self.advantages = advantages

    def sample(self, batch_size: int) -> dataclass:
        indices = np.random.choice(self.length, batch_size, replace=False)
        states, actions, rewards, dones, next_states, action_log_probs, values = zip(
            *[astuple(self.buffer[idx]) for idx in indices])
        return (np.array(states), np.array(actions), np.array(action_log_probs), np.array(values),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states), self.advantages[indices])

    def get_rewards_values(self):
        states, actions, rewards, dones, next_states, action_log_probs, values = zip(
            *[astuple(exp) for exp in self.buffer])
        return np.array(rewards, dtype=np.float32), np.array(values, dtype=np.float32)
