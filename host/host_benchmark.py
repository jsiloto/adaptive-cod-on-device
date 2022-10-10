# Modified from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import argparse
import numpy as np
import torch
import json
import random
import shutil
import time
import traceback
from copy import copy
from torch import jit
from literature_models.model_wrapper import get_all_options, eval_single_model

from literature_models.matsubara2022.wrapper import Matsubara2022

def benchmark_model(model, input_shape, device):

    model.to(device)
    input_shape = (1,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)



# for name, wrapper_class, mode in get_all_options(dummy=False):
#     print(name)
#     wrapper = wrapper_class(mode=mode)
#     wrapper.encoder.set_mode(mode)
#     input_shape = wrapper.get_input_shape()
#     benchmark_model(wrapper.encoder, input_shape,device=torch.device("cuda"))

for name, wrapper_class, mode in get_all_options(dummy=False):
    print(name)
    wrapper = wrapper_class(mode=mode)
    input_shape = wrapper.get_input_shape()
    model_file = wrapper.generate_torchscript("./models")
    model = jit.load(model_file)
    model.set_mode(mode)
    benchmark_model(model, input_shape,device=torch.device("cpu"))
