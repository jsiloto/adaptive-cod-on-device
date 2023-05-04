import os
from typing import List, Tuple

import numpy as np
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
            mode = (4, 4.0)
        self.mode = mode
        encoder_builder = Assine2022BEncoder
        self.encoder = Ensemble(encoder_builder, mode)
        self.jit_encoder = None
        p1 = np.poly1d([-1.97529218, 14.93673171, 3.58359398])
        p2 = np.poly1d([-1.87529218, 14.93673171, 3.58359398])
        p3 = np.poly1d([-1.77529218, 14.93673171, 3.58359398])
        p4 = np.poly1d([-1.67529218, 14.93673171, 3.58359398])

        self.results = {}
        for x in np.arange(1.0, 4.1, 0.1):
            self.results[(1, x)] = (p1(x), 6912.0*x)
            self.results[(2, x)] = (p2(x), 6912.0 * x)
            self.results[(3, x)] = (p3(x), 6912.0 * x)
            self.results[(4, x)] = (p4(x), 6912.0 * x)

        print(self.results)

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


        dict = {'model': self.get_printname(),
                'macs': result[0],
                'params': result[1]}
        reported_results = self.get_reported_results(self.mode)
        dict['map'] = reported_results[0]
        dict['bw'] = reported_results[1]
        return dict

    def get_reported_results(self, mode: Tuple[int, float]) -> (float, float):
        assert mode in self.get_mode_options()
        return self.results[mode]

    def get_best_mode(self, bandwidth, deadline):
        single_compute_time=60.0 #ms

        best = (1, 1.0)
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

        self.jit_encoder.set_mode(10*mode[0])
        return self.jit_encoder
