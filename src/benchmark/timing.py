import time

import numpy as np
import torch


def cpu_exp(model, input_shape, repetitions, device):
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
