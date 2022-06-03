#!/bin/bash

COMMAND="docker run --rm -it \
            --shm-size=32G  \
            -v /work/juliano.siloto/datasets:/work/resource/dataset  \
            -v $PWD:/work -w /work \
            -u $(id -u):$(id -g)  \
            --cap-add LINUX_IMMUTABLE \
            --userns=host  \
            --net=host \
            --privileged -v /dev/bus/usb:/dev/bus/usb \
            --name juliano.siloto.acod-server  \
            juliano.siloto/adaptive_cod_on_device"

#
#COMMAND="docker run --gpus device=0 --rm -it \
#            --shm-size=32G  \
#            -v $PWD:/work -w /work/server \
#            -u $(id -u):$(id -g)  \
#            --cap-add LINUX_IMMUTABLE \
#            --userns=host  \
#            -p 5000:5000 \
#            -e NVIDIA_VISIBLE_DEVICES=\"0\" \
#            -e WANDB_API_KEY=7db49f6a22d959fcba5d6bbaee1f7c28732dd745 \
#            --name juliano.siloto.acod-server.gpu0 \
#            juliano.siloto/adaptive_cod ./setup.sh"

adb root
adb shell setenforce 0
adb shell content insert --uri content://settings/system --bind name:s:accelerometer_rotation --bind value:i:1
adb shell content insert --uri content://settings/system --bind name:s:user_rotation --bind value:i:1
adb shell wm size 800x1200
adb kill-server
#sudo killall -9 bluetoothd
eval "${COMMAND}"
