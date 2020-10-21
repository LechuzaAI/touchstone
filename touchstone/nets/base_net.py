import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = None

    def forward(self, x):
        if self.layers is None:
            raise NotImplementedError("You need to define layers!")

        for layer in self.layers:
            x = layer(x.float())

        return x
