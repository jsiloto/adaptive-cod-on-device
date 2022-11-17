# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import torch
import pandas as pd
from benchmark.timing import cpu_exp
import psutil
import time

from literature_models.base.base_wrapper import BaseWrapper
from literature_models.model_wrapper import get_all_options, eval_single_model, wrapper_dict, build_all_jit_models


def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--cpus', type=int, required=True, help='Number of cpus')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('-switching', action='store_true', default=False, help='Experiment Name')
    return argparser

def benchmark_model_switching(wrapperClass: BaseWrapper, device, name):
    wrapper = wrapperClass()
    results = {}
    for i in range(4):
        for mode in wrapperClass.get_mode_options():
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            al = round(1000 * (time.perf_counter() - bl), 3)
            input_shape = wrapper.get_input_shape()
            timings = np.around(cpu_exp(model, input_shape, 10, device), decimals=2)
            results[name+str(mode)] = {"disk_load": al, "timings": timings.tolist()}
    return results


def main():
    device = "cpu"
    device = torch.device(device)
    args = get_argparser().parse_args()
    torch.set_num_interop_threads(1)
    torch.set_num_threads(args.cpus)

    # Generate all model
    build_all_jit_models()

    results = {}
    if args.switching:
        for name, WrapperClass in wrapper_dict.items():
            if name == "dummy":
                continue
            r = benchmark_model_switching(WrapperClass, device, name)
            results.update(r)

    with open("switching_result.json", "w") as outfile:
        outfile.write(results)


if __name__ == "__main__":
    main()
