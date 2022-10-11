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
from torch.profiler import profile, record_function, ProfilerActivity
from literature_models.matsubara2022.wrapper import Matsubara2022

def benchmark_model(model, input_shape, device):

    model.to(device)
    input_shape = (200,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for i in range(10):
        _ = model(torch.unsqueeze(dummy_input[i], dim=0))
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model(torch.unsqueeze(dummy_input[rep+i], dim=0))
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    #
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(mean_syn)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(torch.unsqueeze(dummy_input[0], dim=0))

        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

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
    # model_file = wrapper.generate_torchscript("./models")
    # model = jit.load(model_file)
    model = wrapper.encoder
    model.set_mode(mode)
    benchmark_model(model, input_shape,device=torch.device("cpu"))
