import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from stable_baselines.common.vec_env import VecEnvWrapper, VecNormalize, DummyVecEnv, SubprocVecEnv


class TimeLimitMask(TimeLimit):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and self._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, env_name):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.dones = []
        self.env_name = env_name
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        self.dones.append(False)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        self.done = done
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    @property
    def global_done(self):
        return all(self.dones)


class VecClip(VecNormalize):
    def __init__(self, *args, **kwargs):
        super(VecClip, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


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
