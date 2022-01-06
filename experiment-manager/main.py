#!/usr/bin/python3

# sudo systemctl start bluetooth
# echo "power on" | bluetoothctl

import matplotlib.pyplot as plt
import collections
import sys
import argparse
import datetime
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from bluetooth import *
from apk_manager import ApkManager
from um25c import UM25C
import adbutils
import os
import re
import json

UM25C_ADDRESS="00:16:A5:00:12:8A"

def measure_power(um25c_device: UM25C, seconds=10):
    data = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        data.append(um25c_device.query())
    return data

def get_argparser():
    argparser = argparse.ArgumentParser(description='Experiment Runner')
    argparser.add_argument('--model', default="", type=str)
    argparser.add_argument('--url', default="", type=str)
    argparser.add_argument('--alpha', default=1.0, type=float)
    argparser.add_argument('--seconds', default=60, type=int)
    return argparser



if __name__ == "__main__":

    args = get_argparser().parse_args()

    os.system('adb root')
    os.system('adb shell setenforce 0')

    # Load Application
    adb_client = adbutils.AdbClient(host="127.0.0.1", port=5037)
    adb_device = adb_client.device()
    apk_manager = ApkManager(adb_device=adb_device, apk_filepath="")

    # Connect to bluetooth device
    um25c = UM25C(UM25C_ADDRESS)

    # Start experiment loop
    apk_manager.start(model=args.model, alpha=args.alpha, url=args.url)
    time.sleep(2)
    apk_manager.clear_logs()
    data = measure_power(um25c_device=um25c, seconds=args.seconds)
    apk_manager.stop()
    experiment_output = apk_manager.parse_outputs()
    x = re.findall("([0-9]+) images", experiment_output)
    num_images = int(x[-1]) - int(x[0])
    print(num_images)


    # Process data
    group = data[0]["group"]
    t = [d['time'] for d in data]
    w = [d['Watts'] for d in data]
    p = [d["6_mWh"] for d in data]

    power_mWh = p[-1] - p[0]
    joules = 0.0
    for i in range(len(t)-1):
        joules += w[i + 1] * (t[i + 1] - t[i]).total_seconds()

    joule_per_image = joules/num_images
    print("Joules1 Total", joules)
    print("Joules1/image", joule_per_image)
    joule_per_image = 3.6*power_mWh / num_images
    print("Joules2 Total", 3.6*power_mWh)
    print("Joules2/image", joule_per_image)


    experiment_data = vars(args)
    experiment_data["joules1"] = joules
    experiment_data["joules1_per_image"] = joule_per_image
    experiment_data["joules2"] = 3.6*power_mWh
    experiment_data["joules2_per_image"] = 3.6*power_mWh / num_images
    experiment_data["num_images"] = num_images
    print(experiment_data)


    experiment_name = "{}_{:3d}_wifi_{}".format(args.model, int(100*args.alpha), not args.url)
    with open(experiment_name+".json", 'w') as f:
        json.dump(experiment_data, f)


    plt.plot(t, w)
    plt.savefig(experiment_name+".png")

