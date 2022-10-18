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
from literature_models.model_wrapper import get_all_options, eval_single_model, wrapper_dict
from torch.profiler import profile, record_function, ProfilerActivity
from literature_models.matsubara2022.wrapper import Matsubara2022


def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--cpus', type=int, required=True, help='Number of cpus')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('-gpu', action='store_true', default=False, help='Experiment Name')
    return argparser



def exp(model, input_shape, repetitions, device):
    input_shape = (repetitions,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    timings = np.zeros((repetitions, 1))
    # MEASURE PERFORMANCE
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(torch.unsqueeze(dummy_input[rep], dim=0))
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    return timings

def benchmark_model_inference(model, input_shape, device):
    warmup_times = 70
    experiment_times = 70
    model.to(device)

    # INIT LOGGERS
    warmup_timings = exp(model, input_shape, warmup_times,  device)
    experiment_timings = exp(model, input_shape, experiment_times,  device)

    return warmup_timings, experiment_timings


def benchmark_model_switching(wrapper: BaseWrapper, device):
    for i in range(4):
        for mode in wrapper.get_mode_options():
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            al = 1000*(time.perf_counter() - bl)
            input_shape = wrapper.get_input_shape()
            timings = exp(model, input_shape, 5, device)
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
    model_path = "./models"
    os.makedirs(model_path, exist_ok=True)
    # for name, wrapper_class, mode in get_all_options(dummy=False):
    #     wrapper = wrapper_class(mode=mode)
    #     model_file = wrapper.generate_torchscript(model_path)
    #
    #
    # for name, wrapper_class in wrapper_dict.items():
    #     model_file = wrapper.generate_torchscript(model_path)
    #     model = jit.load(model_file)
    #     benchmark_model_switching(wrapper, device)

    for name, wrapper_class, mode in get_all_options(dummy=False):
        name = name + "_" + str(mode)
        print(name)
        wrapper = wrapper_class(mode=mode)
        input_shape = wrapper.get_input_shape()
        print(input_shape)
        model_file = wrapper.generate_torchscript(model_path)
        model = jit.load(model_file)
        # model = wrapper.encoder
        model.set_mode(mode)
        warmup_timings, experiment_timings = benchmark_model_inference(model, input_shape, device)
        ms = np.average(experiment_timings)
        print(np.average(experiment_timings), np.std(experiment_timings))
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
