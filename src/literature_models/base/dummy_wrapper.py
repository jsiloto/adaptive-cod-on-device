import os
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from literature_models.base.base_wrapper import BaseWrapper
from literature_models.matsubara2022.encoder import MatsubaraEntropicEncoder


class Dummy(BaseWrapper):

    @classmethod
    def get_mode_options(cls, reduced=False):
        return [1, 2, 3]

    def __init__(self, mode=None):
        self.encoder = None

    def get_printname(self):
        return "dummy"

    def get_input_shape(self) -> (int, int, int):
        return 3, 600, 600

    def generate_torchscript(self, out_dir) -> str:
        return ""

    def generate_metrics(self):
        return {
            'model': "dummy",
            'macs': 0,
            'params': 0,
            'map': 0,
            'bw': 0
        }

    def get_reported_results(self, mode) -> (float, float):
        return (0.0, 0.0)
