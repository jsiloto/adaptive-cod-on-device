# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import os
def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    return argparser


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
import os.path
from literature_models.base.base_wrapper import BaseWrapper
from literature_models.model_wrapper import get_all_options, eval_single_model, wrapper_dict, build_all_jit_models
from torch.profiler import profile, record_function, ProfilerActivity
from literature_models.matsubara2022.wrapper import Matsubara2022
import gdown
import sys
import hashlib

def sha1_checksum(filepath):
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()
    sha1 = hashlib.sha1()

    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
            sha1.update(data)

    return sha1.hexdigest()





def exp(model, input_shape, repetitions, device):
    input_shape = (repetitions,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    timings = np.zeros((repetitions, 1))
    # MEASURE PERFORMANCE
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
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


def benchmark_model_inference(model, input_shape, device):
    warmup_times = 50
    experiment_times = 50
    model.to(device)

    # INIT LOGGERS
    warmup_timings = exp(model, input_shape, warmup_times, device)
    experiment_timings = exp(model, input_shape, experiment_times, device)

    return warmup_timings, experiment_timings


def get_decoder():
    model_filepath = "./models/effd2_decoder.pt"
    sha1sum = "12e0c79f444b78dd38f5ba2bff4cd5062f0b4ccb"
    if not os.path.isfile(model_filepath):
        gdown.download('https://drive.google.com/uc?id=1t4kPsQxV_kgmSL-mXWq4kW5KbG1Sd-U8&export=download',
                       output=model_filepath)

    if sha1_checksum(model_filepath) != sha1sum:
        os.remove(model_filepath)
        print("Download Failed")
        exit(1)

    model = torch.jit.load(model_filepath)
    return model

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




if __name__ == "__main__":
    main()
