import os
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from literature_models.base.base_wrapper import BaseWrapper
from literature_models.matsubara2022.encoder import MatsubaraEntropicEncoder


class Matsubara2022(BaseWrapper):

    @classmethod
    def get_mode_options(cls, reduced=False):
        if reduced:
            return [1]
        else:
            return [1, 2, 3, 4, 5]

    def __init__(self, mode=None):
        self.mode = mode
        self.encoder = MatsubaraEntropicEncoder()
        self.encoders = {}

    def get_printname(self):
        return "matsubara2022_{}".format(self.mode)

    def get_input_shape(self, unsqueeze=False):
        if unsqueeze:
            return 1, 3, 800, 800
        else:
            return 3, 800, 800

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        output_name = "matsubara2022_{}.ptl".format(self.mode)
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self):
        result = get_model_complexity_info(self.encoder, (3, 800, 800),
                                           print_per_layer_stat=False,
                                           as_strings=False)
        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': result[1]}
        reported_results = self.get_reported_results(self.mode)
        dict['map'] = reported_results[0]
        dict['bw'] = reported_results[1]

        # df = pd.DataFrame(dict, index=[0])
        # output_name = "matsubara2022_{}.csv".format(self.mode)
        # out_file = os.path.join(out_dir, output_name)
        # df.to_csv(out_file, index=False)
        return dict

    def get_reported_results(self, mode) -> (float, float):
        assert mode in self.get_mode_options()
        results = {
            1: (36.1, 180e3),
            2: (35.9, 90e3),
            3: (34.0, 23e3),
            4: (29.5, 15e3),
            5: (26.0, 8e3),
        }
        return results[mode]

    def get_encoder(self, mode):
        self.mode = mode
        if mode not in self.encoders:
            print("Model cache miss")
            model_file = f"./models/matsubara2022_{mode}.ptl"
            if not os.path.isfile(model_file):
                print("Model file miss")
                self.generate_torchscript("./models/")

            self.encoders[mode] = torch.jit.load(model_file)
        return self.encoders[mode]