import os

import torch
from torch import nn
from .yolov5_model import parse_model
import os
import yaml
from ptflops import get_model_complexity_info


class LeeYoloV5sEncoder(nn.Module):
    def __init__(self, num_layers: int):
        super(LeeYoloV5sEncoder, self).__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "yolov5s.yaml"), 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
                # print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
        self.model, self.save = parse_model(parsed_yaml, num_layers=num_layers)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=[640, 640], mode='nearest')
        return self.forward_once(x)

    def forward_once(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
        return x

    @torch.jit.export
    def set_mode(self, mode: int) -> int:
        return 0
