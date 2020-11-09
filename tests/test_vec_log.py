import numpy as np
from touchstone.environments.make import make_vec_envs

NUM_ENVS = 2

if __name__ == '__main__':
    env = make_vec_envs("Pendulum-v0", 42, NUM_ENVS)
    np.random.seed(42)

    state = env.reset()
    for i in range(1000):
        actions = env.action_space.sample()
        out = env.step([actions for j in range(NUM_ENVS)])
        # env.render()

    assert abs(env.mean_reward_per_step) < 0.007
    assert len(env.returns) == 1000
