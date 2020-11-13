import torch
import torch.nn as nn


class GaussianLogStd(nn.Module):
    def __init__(self, input_num, output_num, dist_std=1.0):
        super(GaussianLogStd, self).__init__()
        self.mean = nn.Sequential(nn.Linear(input_num, output_num), nn.Tanh())
        self.output_num = output_num
        self.dist_std = dist_std

    def forward(self, x):
        dist_mean = self.mean(x)
        if x.shape[0] == 1:
            dist_var = torch.full((self.output_num,), self.dist_std ** 2, device=dist_mean.device)
            cov_mat = torch.diag(dist_var)
        else:
            dist_var = torch.full((self.output_num,), self.dist_std ** 2, device=dist_mean.device).expand_as(
                dist_mean)
            cov_mat = torch.diag_embed(dist_var)

        return torch.distributions.MultivariateNormal(dist_mean, cov_mat)

    def to(self, *args, **kwargs):
        super(GaussianLogStd, self).to(*args, **kwargs)
        self.mean.to(*args, **kwargs)
