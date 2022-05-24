# adaptive-cod-on-device


```shell
./docker_build.sh
./docker_run.sh
```

```shell
~/Android/Sdk/tools/bin/jobb -d /data/dataset/coco2017/val2017/ -o main.1.org.recod.acod.obb -pn org.recod.acod -pv 1
adb shell mkdir /storage/emulated/0/Android/obb/org.recod.acod/
adb push main.1.org.recod.acod.obb /storage/emulated/0/Android/obb/org.recod.acod/main.1.org.recod.acod.obb
rm  main.1.org.recod.acod.obb
```

```shell
wget https://github.com/steinwurf/adb-join-wifi/releases/download/1.0.1/adb-join-wifi.apk
adb install adb-join-wifi.apk
adb shell am start -n com.steinwurf.adbjoinwifi/.MainActivity -e ssid jsiloto -e password_type WPA -e password senha123
```

```shell
docker exec -it juliano.siloto.acod-server jupyter notebook --allow-root -ip 0.0.0.0
```