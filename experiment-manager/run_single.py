#!/usr/bin/python3

import argparse
import time
import matplotlib.pyplot as plt
from apk_manager import ApkManager
from um25c import UM25C
import adbutils
import os
import re
import json
from tqdm import tqdm
import bluetooth

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


def experiment(seconds: int, model: str, alpha: float, url: str):
    os.system('adb root')
    os.system('adb shell setenforce 0')

    print("Seting Experiment....")

    # Load Application
    adb_client = adbutils.AdbClient(host="127.0.0.1", port=5037)
    adb_device = adb_client.device()
    apk_manager = ApkManager(adb_device=adb_device, apk_filepath="")

    # Connect to bluetooth device
    um25c = UM25C(UM25C_ADDRESS)

    # Start experiment loop
    apk_manager.start(model=model, alpha=alpha, url=url)
    time.sleep(3)  # Warmup
    print("Experiment Start!")
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

    experiment_data = {
        "model": model,
        "alpha": alpha,
        "seconds": seconds,
        "url": url,
        "joules1": joules,
        "joules1_per_image": joules / num_images,
        "joules2": 3.6 * power_mWh,
        "joules2_per_image": 3.6 * power_mWh / num_images,
        "num_images": num_images,
        "compute_latency": seconds / num_images}

    path = './experiment-results'
    os.makedirs(path, exist_ok=True)

    experiment_name = path + "/" + "{}_{:03d}_wifi_{}".format(model, int(100 * alpha), not (not url))
    with open(experiment_name + ".json", 'w') as f:
        json.dump(experiment_data, f)

    plt.plot(t, w)
    plt.savefig(experiment_name + ".png")


def get_argparser():
    argparser = argparse.ArgumentParser(description='Experiment Runner')
    argparser.add_argument('--model', default="", type=str)
    argparser.add_argument('--url', default="", type=str)
    argparser.add_argument('--alpha', default=1.0, type=float)
    argparser.add_argument('--seconds', default=60, type=int)
    return argparser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    experiment(args.seconds, args.model, args.alpha, args.url)
