from dataclasses import dataclass, astuple

import numpy as np

from touchstone.buffers import PPOExperience
from touchstone.buffers.buffer import Buffer


class PPOBuffer(Buffer):

    def append(self, experience: PPOExperience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> dataclass:
        indices = np.random.choice(self.length, batch_size, replace=False)
        states, actions, rewards, dones, next_states, action_log_probs, values = zip(*[astuple(self.buffer[idx]) for idx in indices])
        return (np.array(states), np.array(actions), np.array(action_log_probs), np.array(values),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))
