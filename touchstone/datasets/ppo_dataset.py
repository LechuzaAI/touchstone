from typing import Tuple

from torch.utils.data.dataset import IterableDataset
from touchstone.buffers import PPOBuffer


class PPODataset(IterableDataset):
    def __init__(self, buffer: PPOBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, action_log_probs, values, rewards, dones, new_states, advantages = self.buffer.sample(
            self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], action_log_probs[i], values[i], rewards[i], dones[i], new_states[i], \
                  advantages[i]
