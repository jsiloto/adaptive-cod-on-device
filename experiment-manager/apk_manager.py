import os
import shutil
import adbutils


class ApkManager():

    def __init__(self, adb_device: adbutils.AdbDevice, apk_filepath=""):
        self.adb = adb_device
        self.apk = apk_filepath
        self.application_name = "org.recod.acod" #aapt dump badging self.apk | grep package:\ name


    def start(self, model="", url="", alpha=1.0):
        self.stop()

        self.adb.shell("pm grant {}/ android.permission.READ_EXTERNAL_STORAGE"
                       .format(self.application_name))

        commandline = "am start -n {}/.ExperimentActivity".format(self.application_name)
        commandline += " -e model \"{}\"".format(model)
        commandline += " -e url \"{}\"".format(url)
        commandline += " -ef alpha \"{}\"".format(alpha)
        self.adb.shell(commandline)

    def stop(self):
        self.adb.shell("am force-stop {}".format(self.application_name))

    def install(self):
        self.adb.install(self.apk)


# def get_apk(apk_file):
#     apk_file = "/data/workspace/unicamp/adaptive-cod-on-device/android-device/app/build/outputs/apk/debug/app-debug.apk"
#     local_apk_file = "./experiment.apk"
#     shutil.copy(apk_file, local_apk_file)
#     return local_apk_file
#
#
#     def install()