# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import os
import sys
import inspect
import numpy as np
import torch
import pandas as pd

from timing import gpu_exp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from literature_models.assine_2022b.decoder import get_decoder

def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    return argparser

def benchmark_model_inference(model, input_shape, device):
    warmup_times = 50
    experiment_times = 50
    model.to(device)

    # INIT LOGGERS
    warmup_timings = gpu_exp(model, input_shape, warmup_times, device)
    experiment_timings = gpu_exp(model, input_shape, experiment_times, device)

    return warmup_timings, experiment_timings

def main():
    df = pd.DataFrame(columns=['model', 'ms', 'KB', 'mAP'])
    device = "cuda"
    device = torch.device(device)

    model = get_decoder()
    model.to(device)
    input_shape = (6, 96, 96)

    while (True):
        warmup_timings, experiment_timings = benchmark_model_inference(model, input_shape, device)
        avg, std = np.average(experiment_timings), np.std(experiment_timings)
        print(avg, std)
        if std < avg / 5:
            break
        else:
            print("Unstable experiment -- Rerunning...")
    ms = np.average(experiment_timings)
    print(ms)


if __name__ == "__main__":
    main()
