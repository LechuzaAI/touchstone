import gym
import numpy as np

from touchstone.agents import RandomContinuousAgent

env = gym.make("Pendulum-v0")
np.random.seed(42)
agent = RandomContinuousAgent(env, render=True)
agent.train(500, verbose=True)
