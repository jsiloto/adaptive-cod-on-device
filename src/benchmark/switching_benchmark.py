# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import json
import torch.multiprocessing as mp
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
    argparser.add_argument('--name', type=str, default="Test", help='Model Name')
    return argparser


def benchmark_model_switching(name):
    baseline_ram = psutil.virtual_memory()[3] / 1e9
    print("Baseline Ram USE: {}".format(baseline_ram))
    device = "cpu"
    device = torch.device(device)
    WrapperClass = wrapper_dict[name]
    wrapper = WrapperClass()
    results = {}
    peak_ram_use = 0
    result_list = []
    for i in range(2):
        for mode in wrapper_dict[name].get_mode_options(reduced=False):
            print("Model {}, mode {}".format(name, mode))
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            disk_load = round(1000 * (time.perf_counter() - bl), 3)
            input_shape = wrapper.get_input_shape()
            timings = np.around(cpu_exp(model, input_shape, 10, device), decimals=2).T[0]
            ram_use = psutil.virtual_memory()[3] / 1e9
            if ram_use > peak_ram_use:
                peak_ram_use = ram_use
                print("Peak Ram: {}".format(peak_ram_use-baseline_ram))

            # Test if a cache miss occurred
            if disk_load > 50:
                warmup = timings[0]+timings[1]
                print("Disk_load:{}, Warmup:{}", disk_load, warmup)
                result_list += [{"name": name, "mode": mode, "disk_load": disk_load, "warmup": warmup}]

    results["ram"] = peak_ram_use - baseline_ram
    results["lists"] = result_list
    with open("switching_result_{}.json".format(name), "w") as outfile:
        json.dump(results, outfile)
    return results


def main():
    args = get_argparser().parse_args()
    torch.set_num_interop_threads(1)
    torch.set_num_threads(args.cpus)
    # Generate all model
    benchmark_model_switching(args.name)

if __name__ == "__main__":
    main()
