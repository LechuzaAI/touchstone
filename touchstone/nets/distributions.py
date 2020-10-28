import torch
import torch.nn as nn


class GaussianLogStd(nn.Module):
    def __init__(self, input_num, output_num, action_std=0.5):
        super(GaussianLogStd, self).__init__()
        self.mean = nn.Sequential(nn.Linear(input_num, output_num), nn.Tanh())
        self.action_var = torch.full((output_num,), action_std ** 2)

    def forward(self, x):
        action_mean = self.mean(x)
        action_var = self.action_var.expand_as(action_mean)
        if x.dim() == 2:
            cov_mat = torch.diag(action_var)
        else:
            cov_mat = torch.diag_embed(action_var)
        if action_var.shape[-1] == 1:
            return torch.distributions.Normal(action_mean, action_var)
        else:
            return torch.distributions.MultivariateNormal(action_mean, cov_mat)

    def to(self, *args, **kwargs):
        super(GaussianLogStd, self).to(*args, **kwargs)
        self.mean.to(*args, **kwargs)
        self.action_var.to(*args, **kwargs)
