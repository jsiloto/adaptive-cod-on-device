#!/usr/bin/python3

import argparse
import time
import matplotlib.pyplot as plt
import adbutils
import os
import re
import json
from tqdm import tqdm
import bluetooth

from experiment_manager.apk_manager import ApkManager
from experiment_manager.um25c import UM25C

UM25C_ADDRESS = "00:16:A5:00:12:8A"


def measure_power(um25c_device: UM25C, seconds=10):
    data = []
    start_time = time.time()
    t_end = start_time + seconds
    pbar = tqdm(total=seconds)
    pbar_elapsed = 0
    while time.time() < t_end:
        data.append(um25c_device.query())
        real_elapsed = int(time.time() - start_time)
        if pbar_elapsed < real_elapsed:
            pbar_elapsed = real_elapsed
            pbar.update(1)
    return data


def experiment(seconds: int, model_name: str, model_file: str, mode: float, url: str,
               out_dir: str, norepeat: bool, save: bool=True):
    # Check if experiment exists
    path = out_dir
    os.makedirs(path, exist_ok=True)
    experiment_name = path + "/" + "{}s_{}_{}_wifi_{}"\
        .format(seconds, model_name, mode, not (not url))

    print(experiment_name)
    if norepeat and os.path.exists(experiment_name+".json"):
        exit(0)


    ############### START ##################################


    # os.system('adb root')
    # os.system('adb shell setenforce 0')
    print("Seting Experiment....")
    # Load Application
    adb_client = adbutils.AdbClient(host="127.0.0.1", port=5037)
    adb_device = adb_client.device()
    if len(model_file) > 0:
        print("Installing model: {}".format(model_file))
        adb_device.push(model_file, "/data/user/0/org.recod.acod/files/")
    apk_manager = ApkManager(adb_device=adb_device, apk_filepath="")

    # Connect to bluetooth device
    um25c = UM25C(UM25C_ADDRESS)

    # Start experiment loop
    if "/" in model_file:
        model_file = model_file.split("/")[1]

    pid = ""
    while len(pid) <= 0:
        apk_manager.start(model=model_file, mode=mode, url=url)
        print("Warm Up...")
        time.sleep(3)  # Warmup
        pid = apk_manager.get_pid()

    time.sleep(10)  # Warmup
    print("Experiment Start! PID:{}".format(pid))
    apk_manager.clear_logs()
    data = measure_power(um25c_device=um25c, seconds=seconds)
    apk_manager.stop()
    experiment_output = apk_manager.parse_outputs()
    x = re.findall("([0-9]+) images", experiment_output)
    num_images = int(x[-1]) - int(x[0])

    # Process data
    group = data[0]["group"]
    t = [d['time'] for d in data]
    w = [d['Watts'] for d in data]
    p = [d["6_mWh"] for d in data]

    power_mWh = p[-1] - p[0]
    joules = 0.0
    for i in range(len(t) - 1):
        joules += w[i + 1] * (t[i + 1] - t[i]).total_seconds()

    seconds = (t[-1] - t[0]).total_seconds()

    experiment_data = {
        "model": model_name,
        "mode": mode,
        "seconds": seconds,
        "url": url,
        "joules1": joules,
        "joules1_per_image": joules / num_images,
        "joules2": 3.6 * power_mWh,
        "joules2_per_image": 3.6 * power_mWh / num_images,
        "num_images": num_images,
        "compute_latency": seconds / num_images}

    if save:
        # with open(experiment_name + ".json", 'w') as f:
        #     json.dump(experiment_data, f)
        plt.clf()
        plt.plot(t, w)
        plt.savefig(experiment_name + ".png")

    return experiment_data


def get_argparser():
    argparser = argparse.ArgumentParser(description='Experiment Runner')
    argparser.add_argument('--model', default="", type=str)
    argparser.add_argument('--url', default="", type=str)
    argparser.add_argument('--mode', default=1.0, type=float)
    argparser.add_argument('--seconds', default=60, type=int)
    argparser.add_argument('-no-repeat', action='store_true')
    return argparser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    experiment(args.seconds, args.model, args.mode, args.url, args.no_repeat)
