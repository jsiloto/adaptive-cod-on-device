import os
import shutil


def get_apk(apk_file):
    apk_file = "/data/workspace/unicamp/adaptive-cod-on-device/android-device/app/build/outputs/apk/debug/app-debug.apk"
    local_apk_file = "./experiment.apk"
    shutil.copy(apk_file, local_apk_file)
    return local_apk_file