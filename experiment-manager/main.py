#!/usr/bin/python3

# sudo systemctl start bluetooth
# echo "power on" | bluetoothctl

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
from um25c import UM25C
import adbutils

def measure_power(um25c_device: UM25C, seconds=10):
    data = um25c_device.query()
    return data


if __name__ == "__main__":

    adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
    d = adb.device()
    print(d.serial)  # 获取序列号

    # Parse arguments
    parser = argparse.ArgumentParser(description="CLI for USB Meter")
    parser.add_argument("--addr", dest="addr", type=str, help="Address of USB Meter", required=True)


    args = parser.parse_args()
    print(args.addr)
    try:
        device = UM25C(args.addr)
        while True:
            data = device.query()
            print("%s: %fV %fA %fW" % (data["time"], data["Volts"], data["Amps"], data["Watts"]))

    except KeyboardInterrupt:
        print('Interrupted')
        del device
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)