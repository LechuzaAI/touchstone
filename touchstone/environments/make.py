import gym

from touchstone.environments import DummyVecEnv, ShmemVecEnv, TimeLimitMask, VecClip, VecPyTorch, SubprocVecEnv


def make_env_fn(env_id, seed, rank, log_dir=None, allow_early_resets=False, max_episode_steps=None):
    def make_env():
        env = gym.make(env_id)
        env.seed(seed + rank)
        # TODO: add logging here based on log_dir and allow_early_resets
        # see pytorch-a2c-ppo-acktr-gail repo

        if max_episode_steps is not None:
            env = TimeLimitMask(env, max_episode_steps=max_episode_steps)
        elif str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        return env

    return make_env


def make_envs(env_name, seed, num_processes, gamma, device, max_episode_steps=None):
    envs = [make_env_fn(env_name, seed, i, max_episode_steps=max_episode_steps) for i in range(num_processes)]

    if len(envs) > 1:
        # envs = ShmemVecEnv(envs, context='fork')
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecClip(envs, ret=False)
        else:
            envs = VecClip(envs, gamma=gamma)

    envs = VecPyTorch(envs, device, env_name)

    return envs
