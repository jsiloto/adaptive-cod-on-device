# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import os
import sys
import inspect

from benchmark.timing import cpu_exp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--cpus', type=int, required=True, help='Number of cpus')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('-switching', action='store_true', default=False, help='Experiment Name')
    return argparser
args = get_argparser().parse_args()
os.environ['OMP_NUM_THREADS']=str(args.cpus)
os.environ['MKL_NUM_THREADS']=str(args.cpus)


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

def benchmark_model_switching(wrapperClass: BaseWrapper, device):
    wrapper = wrapperClass()
    for i in range(4):
        for mode in wrapperClass.get_mode_options():
            bl = time.perf_counter()
            model = wrapper.get_encoder(mode)
            model.to(device)
            al = round(1000 * (time.perf_counter() - bl), 3)
            input_shape = wrapper.get_input_shape()
            timings = np.around(cpu_exp(model, input_shape, 10, device), decimals=2)
            print(mode, al, timings.tolist())


def main():
    df = pd.DataFrame(columns=['model', 'ms', 'KB', 'mAP'])
    device = "cpu"
    device = torch.device(device)
    torch.set_num_interop_threads(args.cpus)
    torch.set_num_threads(args.cpus)

    # Generate all model
    build_all_jit_models()

    if args.switching:
        for name, WrapperClass in wrapper_dict.items():
            benchmark_model_switching(WrapperClass, device)
        exit()

    map, kb = wrapper.get_reported_results(mode)
    d = pd.DataFrame({
        'model': name,
        'KB': kb,
        'ms': ms,
        'mAP': map
    }, columns=df.columns, index=[0])
    df = pd.concat([df, d], ignore_index=True)

    df = df.set_index("model")
    print(df)
    filename = "./models/" + args.name + "_cpus{}.csv".format(args.cpus)
    df.to_csv(filename)


if __name__ == "__main__":
    main()
