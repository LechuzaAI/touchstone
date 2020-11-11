from dataclasses import dataclass, astuple
import numpy as np
from touchstone.buffers import PPOExperience, Buffer


class PPOBuffer(Buffer):
    def __init__(self, size: int, num_actors: int):
        super(PPOBuffer, self).__init__(size)
        self.advantages = None
        self.num_actors = num_actors

    def append(self, experience: PPOExperience) -> None:
        self.buffer.append(experience)

    def update_advantages(self, advantages):
        self.advantages = advantages

    def sample(self, batch_size: int) -> dataclass:
        indices = np.random.choice(self.length, batch_size // self.num_actors, replace=False)
        states, actions, rewards, dones, next_states, action_log_probs, values = zip(
            *[astuple(self.buffer[idx]) for idx in indices])
        return (np.array(states).reshape(batch_size, -1),
                np.array(actions).reshape(batch_size, -1),
                np.array(action_log_probs).reshape(batch_size, -1),
                np.array(values).reshape(batch_size, -1),
                np.array(rewards, dtype=np.float32).reshape(batch_size, -1),
                np.array(dones, dtype=np.bool).reshape(batch_size, -1),
                np.array(next_states).reshape(batch_size, -1),
                self.advantages[indices].reshape(batch_size, -1))

    def get_rewards_values(self):
        states, actions, rewards, dones, next_states, action_log_probs, values = zip(
            *[astuple(exp) for exp in self.buffer])
        return np.array(rewards, dtype=np.float32), np.array(values, dtype=np.float32)
