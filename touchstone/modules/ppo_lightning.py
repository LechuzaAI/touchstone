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
        self.explore(self.params.time_steps)
        self.total_reward = 0
        self.episode_reward = 0
        self.iterations = self.params.iterations
        self.time_steps = self.params.time_steps
        self.num_actors = self.params.num_actors
        self.epochs = self.params.epochs
        self.count = 0

    def explore(self, steps: int) -> None:
        for i in range(steps):
            self.agent.play_step(self.actor_critic, deterministic=False, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.actor_critic(x)
        return output

    def on_epoch_start(self) -> None:
        # this is called on every pl epoch, which we call an iteration
        # if we've run at least self.epochs (which are PPO epochs),
        # then run exploration again for self.time_steps
        if self.current_epoch % self.epochs == 0:
            self.explore(self.params.time_steps)
            print("New PPO iteration")

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
