import torch
import torch.nn as nn


class GaussianLogStd(nn.Module):
    def __init__(self, input_num, output_num, action_std=1.0):
        super(GaussianLogStd, self).__init__()
        self.mean = nn.Sequential(nn.Linear(input_num, output_num), nn.Tanh())
        self.output_num = output_num
        self.action_std = action_std

    def forward(self, x):
        action_mean = self.mean(x)
        if x.shape[0] == 1:
            action_var = torch.full((self.output_num,), self.action_std ** 2, device=action_mean.device)
            cov_mat = torch.diag(action_var)
        else:
            action_var = torch.full((self.output_num,), self.action_std ** 2, device=action_mean.device).expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)

        return torch.distributions.MultivariateNormal(action_mean, cov_mat)

    def to(self, *args, **kwargs):
        super(GaussianLogStd, self).to(*args, **kwargs)
        self.mean.to(*args, **kwargs)
        self.action_var.to(*args, **kwargs)
