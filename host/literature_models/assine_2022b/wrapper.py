import os
import torch
from ptflops import get_model_complexity_info

from literature_models.assine_2022b.encoder import Assine2022BEncoder
from literature_models.assine_2022b.ensemble import Ensemble
from literature_models.base.base_wrapper import BaseWrapper


class Assine2022B(BaseWrapper):

    @classmethod
    def get_mode_options(cls):
        # Ensemblesize/numbits
        return [14, 24, 34, 44, 11, 22, 33]

    def __init__(self, mode=None):
        if mode is None:
            mode = 44
        self.mode = mode
        encoder_builder = Assine2022BEncoder
        self.encoder = Ensemble(encoder_builder)

    def get_printname(self):
        return "assine2022b_{}".format(self.mode)

    def get_input_shape(self):
        return 3, 640, 640

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        scripted.eval()
        output_name = "{}.ptl".format(self.get_printname())
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self):
        self.encoder.set_mode(mode=self.mode)
        result = get_model_complexity_info(self.encoder, (3, 640, 640),
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
        results = {
            14: (34.3, 27648.0),
            24: (36.1, 27648.0),
            34: (36.4, 27648.0),
            44: (36.8, 27648.0),

            11: (14.5, 6912.0),
            22: (29.4, 13824.0),
            33: (34.2, 20736.0),

        }
        return results[mode]

