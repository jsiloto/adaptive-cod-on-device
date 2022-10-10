import os
from typing import List

import pandas as pd
import torch
from ptflops import get_model_complexity_info
from literature_models.base.base_wrapper import BaseWrapper
from literature_models.lee2021.encoder import LeeYoloV5sEncoder


class Lee2021(BaseWrapper):

    @classmethod
    def get_mode_options(cls) -> List[int]:
        return [3, 5, 7, 10]

    def __init__(self, mode: int = 3):
        self.mode: int = mode
        self.encoder = LeeYoloV5sEncoder(num_layers=self.mode)

    def get_printname(self):
        return "lee2021_layer_{}".format(self.mode)

    def get_input_shape(self):
        return 3, 640, 640

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        output_name = "lee2021_layer_{}.ptl".format(self.mode)
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self):
        result = get_model_complexity_info(self.encoder, (3, 640, 640),
                                           print_per_layer_stat=False,
                                           as_strings=False)
        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': result[1]}

        reported_results = self.get_reported_results(self.mode)
        dict['map'] = reported_results[0]
        dict['bw'] = reported_results[1]
        # df = pd.DataFrame(dict, index=[0])
        # output_name = "lee2021_layer_{}.csv".format(self.layer)
        # out_file = os.path.join(out_dir, output_name)
        # df.to_csv(out_file, index=False)
        return dict

    def get_reported_results(self, mode) -> (float, int):
        assert mode in self.get_mode_options()
        results = {
            3: (36.8, 30.5e3),
            5: (36.7, 16.2e3),
            7: (36.5, 8.8e3),
            10: (36.4, 4.7e3),
        }
        return results[mode]
