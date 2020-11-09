import torch.nn as nn
from torch.nn import ModuleList, Linear, Tanh
from touchstone.nets import BaseNet, GaussianLogStd


class MLPNet(BaseNet):
    def __init__(self, input_num, output_num, hidden_size=64, dist_output=False):
        super(MLPNet, self).__init__()
        self.layers = ModuleList([Linear(input_num, hidden_size),
                                  Tanh(),
                                  Linear(hidden_size, hidden_size // 2),
                                  Tanh()
                                  ])
        if dist_output:
            self.layers.append(GaussianLogStd(hidden_size // 2, output_num))
        else:
            self.layers.append(nn.Linear(hidden_size // 2, output_num))

        self.input_num = input_num
        self.output_num = output_num
