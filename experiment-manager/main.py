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
from apk_manager import ApkManager
from um25c import UM25C
import adbutils
import os

UM25C_ADDRESS="00:16:A5:00:12:8A"

def measure_power(um25c_device: UM25C, seconds=10):
    data = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        data.append(um25c_device.query())
    return data


if __name__ == "__main__":

    os.system('adb root')
    os.system('adb shell setenforce 0')

    # Load Application
    adb_client = adbutils.AdbClient(host="127.0.0.1", port=5037)
    adb_device = adb_client.device()
    apk_manager = ApkManager(adb_device=adb_device, apk_filepath="")

    # Connect to bluetooth device
    um25c = UM25C(UM25C_ADDRESS)

    # Start experiment loop
    apk_manager.start()
    time.sleep(5)
    data = measure_power(um25c_device=um25c)
    for d in data:
        print(d)

    exit()


    print(d.serial)  # 获取序列号
    d.shell("root")
    time.sleep(3)
    # d.install("../android-device/app/build/outputs/apk/debug/app-debug.apk")


    d.shell("am force-stop com.jsiloto.myapplication")
    d.shell("logcat --clear")
    a = d.shell("am start -n com.jsiloto.myapplication/.DisplayMessageActivity -e TEST \"abcds\"")
    print(a)
    time.sleep(3)
    a = d.shell("logcat -d -e \"ExperimentOutput\"")
    print(a)
    d.shell("am force-stop com.jsiloto.myapplication")
    exit()
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