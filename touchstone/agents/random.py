import itertools
from touchstone.agents import BaseAgent
from touchstone.util import make_continuous_random_policy


class RandomContinuousAgent(BaseAgent):
    def __init__(self, env, render=False, device=None):
        super().__init__(env, policy=make_continuous_random_policy(env.action_space, device))
        self.render = render
        self.device = device

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def explore(self, episode_number):
        state = self.env.reset()
        next_action = self.policy(state)

        for t in itertools.count():
            next_state, reward, done, info = self.env.step(next_action)

            if done:
                break

            next_action = self.policy(next_state)
            # TODO add sample to stats

            if self.render:
                self.env.render()

    def print_info(self, *args, **kwargs):
        pass
