# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import json
import multiprocessing
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


def benchmark_model_switching(name, cpus):
    baseline_ram = psutil.virtual_memory()[3] / 1e9
    print("Ram USE: {}".format(baseline_ram))
    device = "cpu"
    device = torch.device(device)
    torch.set_num_interop_threads(1)
    torch.set_num_threads(cpus)
    wrapper = wrapper_dict[name]
    results = {}
    peak_ram_use = 0
    result_list = []
    for i in range(4):
        for mode in wrapper_dict[name].get_mode_options(reduced=True):
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            al = round(1000 * (time.perf_counter() - bl), 3)
            input_shape = wrapper.get_input_shape()
            timings = np.around(cpu_exp(model, input_shape, 10, device), decimals=2)

            print("Model {}, mode {}".format(name, mode))
            print("Ram USE: {}".format(psutil.virtual_memory()[3] / 1e9))
            ram_use = psutil.virtual_memory()[3] / 1e9
            if ram_use > peak_ram_use:
                peak_ram_use = ram_use

            result_list += [{"name": name, "mode": mode, "disk_load": al, "timings": timings.tolist()}]

    results["ram"] = peak_ram_use - baseline_ram
    results["lists"] = result_list
    with open("switching_result_{}.json".format(name), "w") as outfile:
        json.dump(results, outfile)
    return results


def main():
    args = get_argparser().parse_args()

    # Generate all model
    build_all_jit_models()
    gc.collect()

    for name, _ in wrapper_dict.items():
        if name == "dummy":
            continue
        process = multiprocessing.Process(target=benchmark_model_switching, args=(name, args.cpu))
        process.run()
        process.join()

if __name__ == "__main__":
    main()
