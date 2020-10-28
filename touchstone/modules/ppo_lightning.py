import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from touchstone.agents import PPOAgent
from touchstone.buffers import PPOBuffer
from touchstone.datasets import PPODataset
from touchstone.environments.make import make_envs
from touchstone.nets import ActorCriticNet
import numpy as np


class PPOLightning(pl.LightningModule):
    def __init__(self, params: argparse.Namespace) -> None:
        super(PPOLightning, self).__init__()
        self.params = params
        self.env = make_envs(self.params.env, 42, self.params.num_actors, device=self.device,
                             max_episode_steps=self.params.time_steps)
        observation_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        self.actor_critic = ActorCriticNet(observation_shape, action_shape)
        self.buffer = PPOBuffer(self.params.time_steps * self.params.num_actors)
        self.agent = PPOAgent(self.env, self.buffer)
        self.explore(self.params.time_steps)
        self.total_reward = 0
        self.episode_reward = 0
        self.iterations = self.params.iterations
        self.time_steps = self.params.time_steps
        self.num_actors = self.params.num_actors
        self.epochs = self.params.epochs
        self.count = 0
        self.advantages = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.actor_critic(x)
        return output

    def explore(self, steps: int) -> None:
        for i in range(steps + 1):
            self.agent.explore_step(self.actor_critic, deterministic=False, device=self.device)

    def on_epoch_start(self) -> None:
        # this is called on every pl epoch, which we call an iteration
        # if we've run at least self.epochs (which are PPO epochs),
        # then run we exploration again for self.time_steps
        # Once exploration is done, we need to compute the advantage
        if self.current_epoch % self.epochs == 0:
            self.explore(self.params.time_steps)
            rewards, values = self.buffer.get_rewards_values()
            advantages = self.gae(rewards, values)
            self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            self.buffer.update_advantages(self.advantages)

    def gae(self, rewards, values):
        # TODO look into removing rewards/values for terminal states
        gamma = self.params.gamma
        gae_lambda = self.params.gae_lambda
        advantage = 0
        advantages = [0] * (len(values) - 1)

        for idx in reversed(range(len(values) - 1)):
            delta = rewards[idx] + gamma * values[idx + 1] - values[idx]
            advantage = delta + gamma * gae_lambda * advantage
            advantages[idx] = advantage

        return np.array(advantages)

    def clip_loss(self, batch):
        states, actions, old_action_log_probs, values, rewards, dones, new_states, advantages = batch
        values, action_log_probs = self.agent.evaluate_actions(self.actor_critic, states, actions)
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.params.clip_param, 1.0 + self.params.clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()
        return action_loss

    def training_step(self, batch, nb_batch):
        return self.clip_loss(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.params.lr)
        return [optimizer]

    def __dataloader(self):
        dataset = PPODataset(self.buffer, self.params.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.params.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'
