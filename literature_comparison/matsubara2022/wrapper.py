import sys, os

from matsubara2022.encoder import MatsubaraEntropicEncoder

sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from base_wrapper import BaseWrapper



class Matsubara2022(BaseWrapper):

    @classmethod
    def get_config_options(cls):
        return ["default"]

    def __init__(self, config=None):
        self.layer = config
        self.encoder = MatsubaraEntropicEncoder()

    def get_printname(self):
        return "Matsubara 2022"

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        output_name = "matsubara2022.ptl"
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self, out_dir) -> str:
        result = get_model_complexity_info(self.encoder, (3, 800, 800),
                                           print_per_layer_stat=False,
                                           as_strings=False)
        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': result[1]}
        df = pd.DataFrame(dict, index=[0])
        output_name = "matsubara2022.csv"
        out_file = os.path.join(out_dir, output_name)
        df.to_csv(out_file, index=False)
        return out_file
