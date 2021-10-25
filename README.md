# adaptive-cod-on-device


```shell
docker run --rm -it --gpus all -p 5000:5000 \
 --shm-size=16G -v $PWD:/workspace \
 -w /workspace/server \
 -v /data/dataset/coco2017:/workspace/resource/dataset/coco2017 \
  --name acod-server acod-server ./setup.sh
``

```shell
docker run --rm -it --gpus all \
 --shm-size=16G -v $PWD:/workspace \
 -w /workspace/device \
 -v /data/dataset/coco2017:/workspace/resource/dataset/coco2017 \
  --name acod-device acod-server /bin/bash
``

```shell
adb push android-device/app/src/main/main.1.org.pytorch.demo.objectdetection.obb /storage/emulated/0/Android/obb/org.pytorch.demo.objectdetection
``
