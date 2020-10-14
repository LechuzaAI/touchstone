from gym.wrappers import TimeLimit


class TimeLimitMask(TimeLimit):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and self._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
