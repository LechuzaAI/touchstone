from touchstone.nets import ActorCriticNet


# TODO define a policy base class

class PPOPolicy(object):
    def __init__(self, observation_shape, action_shape):
        self.actor_critic = ActorCriticNet(observation_shape, action_shape)

    def to(self, *args, **kwargs):
        self.actor_critic.to(*args, **kwargs)

    def __call__(self, observation, deterministic=False):
        value, action_distribution = self.actor_critic(observation)
        if deterministic:
            action = action_distribution.mean
        else:
            action = action_distribution.sample()

        action_log_probs = action_distribution.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs
