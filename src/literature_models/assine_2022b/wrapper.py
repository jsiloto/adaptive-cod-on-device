import os
import torch
from ptflops import get_model_complexity_info

from literature_models.assine_2022b.encoder import Assine2022BEncoder
from literature_models.assine_2022b.ensemble import Ensemble
from literature_models.base.base_wrapper import BaseWrapper


class Assine2022B(BaseWrapper):

    @classmethod
    def get_mode_options(cls, reduced=False):
        # Ensemblesize/numbits
        if reduced:
            return [11, 22, 33, 44]
        else:
            return [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44]
        # return [14, 24, 34, 44]




    def __init__(self, mode=None):
        if mode is None:
            mode = 44
        self.mode = mode
        encoder_builder = Assine2022BEncoder
        self.encoder = Ensemble(encoder_builder, mode)
        self.jit_encoder = None
        self.results = {
            11: (14.5, 6912.0),
            12: (27.7, 13824.0),
            13: (32.0, 20736.0),
            14: (34.3, 27648.0),

            21: (16.5, 6912.0),
            22: (29.4, 13824.0),
            23: (34.1, 20736.0),
            24: (36.1, 27648.0),

            31: (16.4, 6912.0),
            32: (29.4, 13824.0),
            33: (34.2, 20736.0),
            34: (36.4, 27648.0),

            41: (16.6, 6912.0),
            42: (29.7, 13824.0),
            43: (34.6, 20736.0),
            44: (36.8, 27648.0),
        }

    def get_printname(self):
        return "assine2022b_{}".format(self.mode)

    def get_input_shape(self, unsqueeze=False):
        if unsqueeze:
            return 1, 3, 384, 384
        else:
            return 3, 384, 384

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        scripted.eval()
        output_name = "assine2022b.ptl".format(self.get_printname())
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self):
        self.encoder.set_mode(mode=self.mode)
        result = get_model_complexity_info(self.encoder, self.get_input_shape(),
                                           print_per_layer_stat=False,
                                           as_strings=False)
        print(result)
        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': result[1]}
        reported_results = self.get_reported_results(self.mode)
        dict['map'] = reported_results[0]
        dict['bw'] = reported_results[1]
        return dict

    def get_reported_results(self, mode: int) -> (float, float):
        assert mode in self.get_mode_options()
        return self.results[mode]

    def get_best_mode(self, bandwidth, deadline):
        single_compute_time=60.0 #ms

        best = 11
        best_map = 14.5
        for mode, (map, kb) in self.results.items():
            compute_time = single_compute_time*(mode/10)
            transmit_time = kb/bandwidth
            if compute_time+transmit_time < 0.8*deadline:
                if map > best_map:
                    best = mode
                    best_map = map

        return best



    def get_encoder(self, mode, force=False):
        model_file = "./models/assine2022b.ptl"
        if self.jit_encoder is None:
            print("Model cache miss")
            if not os.path.isfile(model_file) or force:
                print("Model file miss")
                self.generate_torchscript("./models/")
            self.jit_encoder = torch.jit.load("./models/assine2022b.ptl")

        self.jit_encoder.set_mode(mode)
        return self.jit_encoder
