import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from touchstone.agents import PPOAgent
from touchstone.buffers import PPOBuffer
from touchstone.datasets import PPODataset
from touchstone.environments.make import make_vec_envs
from touchstone.nets import ActorCriticNet
import numpy as np


class PPOLightning(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super(PPOLightning, self).__init__()
        self.hparams = hparams
        self.env = make_vec_envs(self.hparams.env, 42, self.hparams.num_actors)
        observation_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        self.actor_critic = ActorCriticNet(observation_shape, action_shape, action_std=self.hparams.action_std)
        self.actor_critic_old = ActorCriticNet(observation_shape, action_shape, action_std=self.hparams.action_std)
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        self.buffer = PPOBuffer(self.hparams.time_steps, self.hparams.num_actors)
        self.agent = PPOAgent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.iterations = self.hparams.iterations
        self.time_steps = self.hparams.time_steps
        self.num_actors = self.hparams.num_actors
        self.epochs = self.hparams.epochs
        self.count = 0
        self.advantages = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.actor_critic(x)
        return output

    def explore(self, steps: int) -> None:
        for i in range(steps + 1):
            self.agent.explore_step(self.actor_critic_old, device=self.device)

    def on_train_start(self) -> None:
        self.env.reset()
        self.explore(self.hparams.time_steps)

    def on_epoch_start(self) -> None:
        # this is called on every pl epoch, which we call an iteration
        # if we've run at least self.epochs (which are PPO epochs),
        # then run we exploration again for self.time_steps
        # Once exploration is done, we need to compute the advantage
        if self.current_epoch % self.epochs == 0:
            self.explore(self.hparams.time_steps)
            rewards, values = self.buffer.get_rewards_values()
            advantages = self.gae(rewards, values)
            self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            self.buffer.update_advantages(self.advantages)

    def on_epoch_end(self) -> None:
        if self.current_epoch % self.epochs == 0:
            self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

    def gae(self, rewards, values):
        # TODO look into removing rewards/values for terminal states
        gamma = self.hparams.gamma
        gae_lambda = self.hparams.gae_lambda
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
        ratio = torch.exp(action_log_probs.unsqueeze(dim=1) - old_action_log_probs).squeeze()
        advantages = advantages.squeeze()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.hparams.clip_param, 1.0 + self.hparams.clip_param) * advantages
        action_loss = -torch.min(surr1, surr2)
        return action_loss.mean()

    def training_step(self, batch, nb_batch):
        loss = self.clip_loss(batch)
        self.log('mean_reward_per_step', self.env.mean_reward_per_step[0], on_step=True, on_epoch=True, logger=True)
        self.log('episode_reward', self.env.episode_rewards[-1][0], on_step=True, on_epoch=True, logger=True)
        self.log('training_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        return [optimizer]

    def __dataloader(self):
        dataset = PPODataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'
