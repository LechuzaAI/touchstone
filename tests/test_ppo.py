import gym

from touchstone.agents import PPOAgent

NUM_ENV_STEPS = 1000000

env = gym.make("Pendulum-v0")

ppo_params = {"T": 512, "N": 1, "clip": 0.2, "batch_size": 64, "lr": 3e-4, "eps": 1e-5}

agent = PPOAgent(env, params=ppo_params)
agent.train(NUM_ENV_STEPS, verbose=True)
