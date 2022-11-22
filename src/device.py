import argparse
import os
import shutil
import time
import jsonlines
import torch

from btcomm.bt import BTClient
from literature_models.assine_2022b.wrapper import Assine2022B


def get_argparser():
    argparser = argparse.ArgumentParser(description='Device Side Full System')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('--seconds', type=int, default=100, help='Total Runtime')
    argparser.add_argument('--deadline', type=int, default=200, help='Frame Deadline')
    argparser.add_argument("--addr", required=True, type=str, help="If client, please specify target connection")
    argparser.add_argument('--mode', type=int, default=-1, help='(11, 12, .. 44) or -1 (dynamic)')
    #argparser.add_argument('-multithread', type=str, default=10, help='separate compute/communicate threads')
    return argparser

def main():
    # Load Model
    args = get_argparser().parse_args()
    mode = 11
    wrapper = Assine2022B()
    model = wrapper.get_encoder(mode, force=True)
    input_shape = wrapper.get_input_shape(unsqueeze=True)
    device = torch.device("cpu")

    result_file = "./bt_results.jsonl"
    shutil.rmtree(result_file, ignore_errors=True)


# Connect BT
    bt_client = BTClient(args.addr)
    start = time.time()
    bw_moving_average = 200
    writer = jsonlines.open('output.jsonl', mode='w')
    error = 0.0
    previous_error = 0.0
    target = args.deadline
    while time.time() - start < args.seconds:
        e2e_start = time.time()
        dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
        model(dummy_input)
        model_time = time.time()-e2e_start
        print("Model time", model_time)
        kbs = wrapper.get_reported_results(mode)[1]/1000.0
        rtt_time = bt_client.send(kbs)
        expected_server_time = 0.060
        rtt_time -= expected_server_time
        bandwidth = kbs/rtt_time
        alpha = 0.4
        last_bw = bw_moving_average
        bw_moving_average = alpha*bandwidth + (1-alpha)*last_bw
        e2e_end = time.time()
        e2e = e2e_end - e2e_start
        expected_bw = bw_moving_average + (bw_moving_average - last_bw)/4


        print("End-to-End Time:{} / Deadline: {} / BW:{}".format(e2e*1000, args.deadline, expected_bw))

        if args.mode == -1:
            mode = wrapper.get_best_mode(bw_moving_average, args.deadline)
        else:
            mode = args.mode
        model = wrapper.get_encoder(mode)
        mAP, kbs = wrapper.results[mode]
        print("Setting Mode/MaP: {}/{}".format(mode, mAP))


        results ={
            "bw": bw_moving_average,
            "e2e": 1000*e2e,
            "mode": mode,
            "map": mAP,
            "model_time": model_time,
            "kbs": kbs/1000,
            "deadline": args.deadline,
            "time": time.time()
        }

        writer.write(results)


if __name__ == "__main__":
    main()
