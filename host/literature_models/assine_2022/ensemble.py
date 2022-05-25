import torch
from torch import nn
from typing import List, Tuple


class Ensemble(nn.Module):
    def __init__(self, encoder_builder):
        super().__init__()
        self.size: int = 4
        self.scale_factor: float = 0.5

        print("Building Ensemble of size {}, scale factor {}"
              .format(self.size, self.scale_factor))

        self.models = nn.ModuleList([
            encoder_builder()
            for i in range(self.size)
        ])

    @torch.jit.export
    def set_mode(self, mode: int):
        self.size = int(mode / 10)
        # print("Setting size: {}".format(self.size))

    def forward(self, x):
        output_list = []
        x_ = torch.nn.functional.interpolate(
            x, scale_factor=[self.scale_factor, self.scale_factor], mode='nearest')
        for i, m in enumerate(self.models):
            if i < self.size:
                y = m.forward(x_)
                y = y*(2**(1-i))
                output_list.append(y)

        y = torch.sum(torch.stack(output_list), dim=0)
        return y

