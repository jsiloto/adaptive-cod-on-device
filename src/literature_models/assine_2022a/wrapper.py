import os
import torch
from ptflops import get_model_complexity_info
from literature_models.assine_2022a.encoder import Assine2022AEncoder
from literature_models.base.base_wrapper import BaseWrapper
from literature_models.common.analyzer.hooks import custom_module_mapping

class Assine2022A(BaseWrapper):

    @classmethod
    def get_mode_options(cls, reduced=False):
        # alpha
        return [25, 50, 75, 100]

    def __init__(self, mode=None):
        if mode is None:
            mode = 44
        self.mode = mode
        self.encoder = Assine2022AEncoder()
        self.jit_encoder = None

    def get_printname(self):
        return "assine2022a_{}".format(self.mode)

    def get_input_shape(self, unsqueeze=False):
        if unsqueeze:
            return 1, 3, 768, 768
        else:
            return 3, 768, 768

    def generate_torchscript(self, out_dir) -> str:
        scripted = torch.jit.script(self.encoder)
        scripted.eval()
        output_name = "assine2022a.ptl"
        out_file = os.path.join(out_dir, output_name)
        scripted._save_for_lite_interpreter(out_file)
        return out_file

    def generate_metrics(self):
        self.encoder.set_mode(mode=self.mode)
        result = get_model_complexity_info(self.encoder, (3, 768, 768),
                                           print_per_layer_stat=False,
                                           as_strings=False,
                                           custom_modules_hooks=custom_module_mapping)

        a = 0
        for m in self.encoder.modules():
            if hasattr(m, "slimmable_params"):
                a += m.slimmable_params

        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': a+result[1]-173088}
        reported_results = self.get_reported_results(self.mode)
        dict['map'] = reported_results[0]
        dict['bw'] = reported_results[1]
        return dict

    def get_reported_results(self, mode: int) -> (float, float):
        assert mode in self.get_mode_options()
        results = {
            25: (31.65, 110e3),
            50: (37.84, 220e3),
            75: (39.88, 330e3),
            100: (39.55, 440e3),
        }

        return results[mode]

    def get_encoder(self, mode):
        if self.jit_encoder is None:
            print("Model cache miss")
            self.jit_encoder = torch.jit.load("./models/assine2022a.ptl")

        self.jit_encoder.set_mode(mode)
        return self.jit_encoder