# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import json
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
# from memory_profiler import memory_usage
import gc


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
        for mode in wrapperClass.get_mode_options(reduced=True):
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            al = round(1000 * (time.perf_counter() - bl), 3)
            input_shape = wrapper.get_input_shape()
            timings = np.around(cpu_exp(model, input_shape, 10, device), decimals=2)
            results[name+str(mode)] = {"disk_load": al, "timings": timings.tolist()}
            print("Model {}, mode {}".format(name, mode))
            print("Ram USE: {}".format(psutil.virtual_memory()[3] / 1e9))

    del wrapper
    return results


def main():
    device = "cpu"
    device = torch.device(device)
    args = get_argparser().parse_args()
    torch.set_num_interop_threads(1)
    torch.set_num_threads(args.cpus)

    baseline_ram = psutil.virtual_memory()[3]/1e9


    # Generate all model
    build_all_jit_models()
    gc.collect()
    results = {"baseline_ram": baseline_ram}

    for name, WrapperClass in wrapper_dict.items():
        gc.collect()
        print(gc.get_stats())
        if name == "dummy":
            continue
        print("Ram USE: {}".format(psutil.virtual_memory()[3]/1e9))
        r = benchmark_model_switching(WrapperClass, device, name)
        results.update(r)

    with open("switching_result.json", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
