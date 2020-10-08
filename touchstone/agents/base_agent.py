class BaseAgent(object):
    def __init__(self, env, validate_action_fn=None, policy=None):
        self.env = env
        self.validate_action_fn = validate_action_fn
        self.policy = policy

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def explore(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, num_episodes, verbose=False):
        for episode_index in range(num_episodes):
            self.explore(episode_index)
            if verbose:
                self.print_info(episode_index, num_episodes)

    def print_info(self, *args, **kwargs):
        raise NotImplementedError
