from collections import deque
from dataclasses import dataclass
import numpy as np

from touchstone.buffers import Experience


class Buffer:
    def __init__(self, size: int) -> None:
        self.length = size
        self.buffer = deque(maxlen=size)

    def __len__(self) -> int:
        return self.length

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> dataclass:
        indices = np.random.choice(self.length, batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))
