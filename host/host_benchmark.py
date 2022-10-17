# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
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
from literature_models.model_wrapper import get_all_options, eval_single_model
from torch.profiler import profile, record_function, ProfilerActivity
from literature_models.matsubara2022.wrapper import Matsubara2022


def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--cpus', type=int, required=True, help='Number of cpus')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('-gpu', action='store_true', default=False, help='Experiment Name')
    return argparser


def benchmark_model(model, input_shape, device):
    warmup_times = 50
    experiment_times = 50
    model.to(device)
    input_shape = (warmup_times + experiment_times,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def exp(repetitions):
        timings = np.zeros((experiment_times, 1))
        # MEASURE PERFORMANCE
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

    warmup_timings = exp(warmup_times)
    experiment_timings = exp(experiment_times)

    return warmup_timings, experiment_timings


# for name, wrapper_class, mode in get_all_options(dummy=False):
#     print(name)
#     wrapper = wrapper_class(mode=mode)
#     wrapper.encoder.set_mode(mode)
#     input_shape = wrapper.get_input_shape()
#     benchmark_model(wrapper.encoder, input_shape,device=torch.device("cuda"))


def main():
    args = get_argparser().parse_args()
    df = pd.DataFrame(columns=['model', 'ms', 'KB', 'mAP'])
    device = "cpu"
    if args.gpu:
        device = "cuda"

    torch.set_num_interop_threads(args.cpus)
    torch.set_num_threads(args.cpus)

    for name, wrapper_class, mode in get_all_options(dummy=False):
        name = name + "_" + str(mode)
        print(name)
        wrapper = wrapper_class(mode=mode)
        input_shape = wrapper.get_input_shape()
        print(input_shape)
        model_file = wrapper.generate_torchscript("./models")
        model = jit.load(model_file)
        # model = wrapper.encoder
        model.set_mode(mode)
        warmup_timings, experiment_timings = benchmark_model(model, input_shape, device=torch.device(device))
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
