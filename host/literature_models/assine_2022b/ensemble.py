import torch
from torch import nn
from typing import List, Tuple

from literature_models.common.efficientdet.preprocess import preprocess_for_torchscript


class Ensemble(nn.Module):
    def __init__(self, encoder_builder, mode):
        super().__init__()
        self.size: int = mode // 10
        self.scale_factor: float = 0.5

        # print("Building Ensemble of size {}, scale factor {}"
        #       .format(self.size, self.scale_factor))

        self.models = nn.ModuleList([
            encoder_builder()
            for i in range(self.size)
        ])

    @torch.jit.export
    def set_mode(self, mode: int):
        self.size = int(mode // 10)
        # print("Setting size: {}".format(self.size))

    def forward(self, x):
        # x = preprocess_for_torchscript(x, max_size=768)
        # output_list = []
        x_ = torch.nn.functional.interpolate(
            x, scale_factor=[self.scale_factor, self.scale_factor], mode='nearest')

        y = self.models[0].forward(x_)
        for i, m in enumerate(self.models[1:]):
            if i < self.size-1:
                y_ = m.forward(x_)
                y_ = y_ * (2 ** (1 - i))
                y = y + y_

        # y = torch.sum(torch.stack(output_list), dim=0)
        return y
