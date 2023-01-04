# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import os
import sys
import inspect

from timing import cpu_exp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--cpus', type=int, required=True, help='Number of cpus')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('-full', action="store_true", help='Run all modes')
    return argparser

args = get_argparser().parse_args()
# os.environ['OMP_NUM_THREADS']=str(args.cpus)
# os.environ['MKL_NUM_THREADS']=str(args.cpus)


import numpy as np
import torch
import pandas as pd
from literature_models.model_wrapper import get_all_options, eval_single_model, wrapper_dict, build_all_jit_models



def benchmark_model_inference(model, input_shape, device):
    warmup_times = 10
    experiment_times = 10
    model.to(device)

    # INIT LOGGERS
    warmup_timings = cpu_exp(model, input_shape, warmup_times, device)
    experiment_timings = cpu_exp(model, input_shape, experiment_times, device)

    return warmup_timings, experiment_timings



def main():
    df = pd.DataFrame(columns=['model', 'ms', 'KB', 'mAP'])
    device = "cpu"
    device = torch.device(device)
    torch.set_num_interop_threads(args.cpus)
    torch.set_num_threads(1)

    # Generate all model
    build_all_jit_models()

    for name, wrapper_class, mode in get_all_options(dummy=False, reduced=(not args.full)):
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
