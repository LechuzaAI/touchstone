# import torch
# from torch import optim
# from touchstone.agents import BaseAgent
# from touchstone.buffers import PPOExperienceBuffer
# from touchstone.policies.make import make_ppo_policy
# from touchstone.utilities import generalized_advantage_estimate
#
#
# class PPOAgent(BaseAgent):
#     def __init__(self, env, params, render=False, device="cpu"):
#         super(PPOAgent, self).__init__(env, policy=make_ppo_policy(env.observation_space.shape,
#                                                                    env.action_space.shape))
#         self.params = params
#         self.device = device
#         self.render = render
#         self.experience_buffer = PPOExperienceBuffer(self.params['T'],
#                                                      self.params['N'],
#                                                      self.params['M'],
#                                                      env.observation_space.shape,
#                                                      env.action_space.shape)
#         if self.device is not None:
#             self.policy.to(self.device)
#
#         self.optimizer = optim.Adam(self.policy.actor_critic.parameters(), lr=self.params['lr'], eps=self.params['eps'])
#
#     def train(self, iterations, verbose=False):
#         self.experience_buffer.push(self.env.reset())
#         for i in range(iterations):
#             self.explore(self.params["T"])
#             self.compute_advantages()
#             self.optimize()
#             if verbose:
#                 self.print_info(i, iterations)
#
#     def optimize(self):
#         # TODO use pytorch lightning here
#         for epoch in range(self.params["K"]):
#             for sample_batch in self.experience_buffer:
#                 print(sample_batch)
#
#     def compute_advantages(self):
#         generalized_advantage_estimate(self.experience_buffer, self.params['gamma'], self.params['lambda'],
#                                        self.params['T'])
#
#     def explore(self, T):
#         for t in range(T):
#             with torch.no_grad():
#                 value, action, action_log_probs = self.policy(self.experience_buffer[t])
#
#             observation, reward, done, info = self.env.step(action)
#             self.experience_buffer.insert(observation, action, action_log_probs, value, reward)
#
#     def print_info(self, *args, **kwargs):
#         pass
#
#     def load(self, *args, **kwargs):
#         pass
#
#     def save(self, *args, **kwargs):
#         pass
