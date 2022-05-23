import sys, os

sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from base_wrapper import BaseWrapper
from lee2021.encoder import LeeYoloV5sEncoder


class Lee2021(BaseWrapper):

    @classmethod
    def get_config_options(cls):
        return ["3", "5", "7", "10"]

    def __init__(self, config=None):
        if config is None:
            config = "3"
        self.layer = config
        self.encoder = LeeYoloV5sEncoder(num_layers=int(self.layer))

    def get_printname(self):
        return "Lee2021 - Split @ Layer {}".format(self.layer)

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        output_name = "lee2021_layer_{}.ptl".format(self.layer)
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self, out_dir) -> str:
        result = get_model_complexity_info(self.encoder, (3, 640, 640),
                                           print_per_layer_stat=False,
                                           as_strings=False)
        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': result[1]}
        df = pd.DataFrame(dict, index=[0])
        output_name = "lee2021_layer_{}.csv".format(self.layer)
        out_file = os.path.join(out_dir, output_name)
        df.to_csv(out_file, index=False)
        return out_file
