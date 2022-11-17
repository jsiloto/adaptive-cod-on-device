import argparse
import os
import time

import torch

from btcomm.bt import BTClient
from literature_models.assine_2022b.wrapper import Assine2022B


def get_argparser():
    argparser = argparse.ArgumentParser(description='Device Side Full System')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('--seconds', type=int, default=10, help='Total Runtime')
    argparser.add_argument('--mode', type=str, default=10, help='Mode of operation (11 12 .. 44) or dynamic')
    argparser.add_argument("--addr", required=True, type=str, help="If client, please specify target connection")
    #argparser.add_argument('-multithread', type=str, default=10, help='separate compute/communicate threads')
    return argparser

def main():
    # Load Model
    args = get_argparser().parse_args()
    mode = int(args.mode)
    wrapper = Assine2022B()
    model = wrapper.get_encoder(mode)
    input_shape = wrapper.get_input_shape()
    device = torch.device("cpu")

# Connect BT
    bt_client = BTClient(args.addr)
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    start = time.time()
    while time.time() - start < args.seconds:
        model(dummy_input)
        kbs = wrapper.get_reported_results(mode)[1]/1000.0
        bt_client.send(kbs)


if __name__ == "__main__":
    main()
