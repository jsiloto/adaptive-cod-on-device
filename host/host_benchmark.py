# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import os

import numpy as np
import torch
import pandas as pd
import json
import random
import shutil
import time
import traceback
from copy import copy
from torch import jit
import time

from literature_models.base.base_wrapper import BaseWrapper
from literature_models.model_wrapper import get_all_options, eval_single_model, wrapper_dict, build_all_jit_models
from torch.profiler import profile, record_function, ProfilerActivity
from literature_models.matsubara2022.wrapper import Matsubara2022


def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--cpus', type=int, required=True, help='Number of cpus')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('-gpu', action='store_true', default=False, help='Experiment Name')
    argparser.add_argument('-switching', action='store_true', default=False, help='Experiment Name')
    return argparser


def exp(model, input_shape, repetitions, device):
    input_shape = (repetitions,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    timings = np.zeros((repetitions, 1))
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            start = time.time()
            _ = model(torch.unsqueeze(dummy_input[rep], dim=0))
            end = time.time()
            # WAIT FOR GPU SYNC
            curr_time = end-start
            timings[rep] = curr_time*1000
    return timings


def benchmark_model_inference(model, input_shape, device):
    warmup_times = 50
    experiment_times = 50
    model.to(device)

    # INIT LOGGERS
    warmup_timings = exp(model, input_shape, warmup_times, device)
    experiment_timings = exp(model, input_shape, experiment_times, device)

    return warmup_timings, experiment_timings


def benchmark_model_switching(wrapperClass: BaseWrapper, device):
    wrapper = wrapperClass()
    for i in range(4):
        for mode in wrapperClass.get_mode_options():
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            al = round(1000 * (time.perf_counter() - bl), 3)
            input_shape = wrapper.get_input_shape()
            timings = np.around(exp(model, input_shape, 10, device), decimals=2)
            print(mode, al, timings.tolist())


def main():
    args = get_argparser().parse_args()
    df = pd.DataFrame(columns=['model', 'ms', 'KB', 'mAP'])
    device = "cpu"
    if args.gpu:
        device = "cuda"
    device = torch.device(device)
    torch.set_num_interop_threads(args.cpus)
    torch.set_num_threads(args.cpus)

    # Generate all model
    build_all_jit_models()

    if args.switching:
        for name, WrapperClass in wrapper_dict.items():
            benchmark_model_switching(WrapperClass, device)
        exit()

    for name, wrapper_class, mode in get_all_options(dummy=False):
        wrapper = wrapper_class(mode=mode)
        name = name + "_" + str(mode)
        model = wrapper.get_encoder(mode)
        input_shape = wrapper.get_input_shape()
        print(name, input_shape)

        while (True):
            warmup_timings, experiment_timings = benchmark_model_inference(model, input_shape, device)
            avg, std = np.average(experiment_timings), np.std(experiment_timings)
            print(avg, std)
            if std < avg / 5:
                break
            else:
                print("Unstable experiment -- Rerunning...")
        ms = np.average(experiment_timings)
        map, kb = wrapper.get_reported_results(mode)
        d = {
            'model': name,
            'KB': kb,
            'ms': ms,
            'mAP': map
        }
        df = df.append(d, ignore_index=True)

    df = df.set_index("model")
    print(df)
    filename = "./models/" + args.name + "_cpus{}.csv".format(args.cpus)
    df.to_csv(filename)


if __name__ == "__main__":
    main()
