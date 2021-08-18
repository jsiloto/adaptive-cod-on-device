# adaptive-cod-on-device


```shell
docker run --rm -it --gpus all -p 5000:5000 \
 --shm-size=16G -v $PWD:/workspace \
 -w /workspace/server \
 -v /data/dataset/coco2017:/workspace/resource/dataset/coco2017 \
  --name acod-server acod-server ./setup.sh
``
