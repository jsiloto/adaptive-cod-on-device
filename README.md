# adaptive-cod-on-device


```shell
docker run --rm -it --gpus all -p 5000:5000 \
 --shm-size=16G -v $PWD:/workspace \
 -w /workspace/server \
 -v /data/dataset/coco2017:/workspace/resource/dataset/coco2017 \
  --name acod-server acod-server ./setup.sh
```

```shell
docker run --rm -it --gpus all \
 --shm-size=16G -v $PWD:/workspace \
 -w /workspace/device \
 -v /data/dataset/coco2017:/workspace/resource/dataset/coco2017 \
  --name acod-device acod-server /bin/bash
```

```shell
~/Android/Sdk/tools/bin/jobb -d /data/dataset/coco2017/val2017/ -o main.1.org.recod.acod.obb -pn org.recod.acod -pv 1
adb shell mkdir /storage/emulated/0/Android/obb/org.recod.acod/
adb push main.1.org.recod.acod.obb /storage/emulated/0/Android/obb/org.recod.acod/
rm  main.1.org.recod.acod.obb
```
