import argparse
import gym
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from touchstone.agents import PPOAgent
from touchstone.buffers import PPOBuffer
from touchstone.datasets import PPODataset
from touchstone.nets import ActorCriticNet


class PPOLightning(pl.LightningModule):
    def __init__(self, params: argparse.Namespace) -> None:
        super(PPOLightning, self).__init__()
        self.params = params
        self.env = gym.make(self.params.env)
        observation_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        self.actor_critic = ActorCriticNet(observation_shape, action_shape)
        self.buffer = PPOBuffer(self.params.time_steps * self.params.num_actors)
        self.agent = PPOAgent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.explore(self.params.time_steps)

    def explore(self, steps: int) -> None:
        for i in range(steps):
            self.agent.play_step(self.actor_critic, deterministic=False, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.actor_critic(x)
        return output

    def ppo_loss(self, batch):
        pass

    def training_step(self, batch, nb_batch):
        pass

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
