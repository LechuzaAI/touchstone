import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ParameterVector(nn.Module):
    def __init__(self, num_elements):
        super().__init__()
        self.__bias = nn.Parameter(torch.zeros(num_elements).unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self.__bias.t().view(1, -1)
        else:
            bias = self.__bias.t().view(1, -1, 1, 1)

        return bias


class LogStdGaussian(nn.Module):
    def __init__(self, input_num, output_num):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.mean = nn.Sequential(init_(nn.Linear(input_num, output_num)), nn.Tanh())
        self.log_std = ParameterVector(output_num)

    def forward(self, x):
        action_mean = self.mean(x)
        action_log_std = self.log_std(x).exp()
        return torch.distributions.Normal(action_mean, action_log_std)

    def to(self, *args, **kwargs):
        super(LogStdGaussian, self).to(*args, **kwargs)
        self.mean.to(*args, **kwargs)
        self.log_std.to(*args, **kwargs)
