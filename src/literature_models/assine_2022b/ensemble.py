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

        a = [torch.jit.fork(m, x) for m in self.models]
        outs = [b.wait() for b in a]
        y = outs[0]
        for i, o in enumerate(outs[1:]):
            y = y + o * (2 ** (-i))

        # y = torch.sum(torch.stack(output_list), dim=0)
        return y
