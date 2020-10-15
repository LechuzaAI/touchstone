import gym
import torch

from touchstone.agents import PPOAgent
from touchstone.environments.make import make_envs

NUM_ENV_STEPS = 1000000
SEED = 42
N = 1
T = 100
GAMMA = 0.99
GPUS = 1
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and GPUS > 0) else "cpu")
MAX_EPISODE_STEPS = 200

# env = gym.make("Pendulum-v0")
env = make_envs("Pendulum-v0", SEED, N, GAMMA, DEVICE, max_episode_steps=MAX_EPISODE_STEPS)

ppo_params = {"T": T, "N": N, "clip": 0.2, "batch_size": 64, "lr": 3e-4, "eps": 1e-5, "gamma": GAMMA, "lambda": 0.99}

agent = PPOAgent(env, params=ppo_params)
agent.train(NUM_ENV_STEPS, verbose=True)
