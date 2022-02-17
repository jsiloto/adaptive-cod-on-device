#!/bin/bash

COMMAND="docker run --rm -it \
            --shm-size=32G  \
            -v /work/juliano.siloto/datasets:/work/resource/dataset  \
            -v $PWD:/work -w /work/server \
            -u $(id -u):$(id -g)  \
            --cap-add LINUX_IMMUTABLE \
            --userns=host  \
            -p 5000:5000 \
            --name juliano.siloto.acod-server  \
            juliano.siloto/adaptive_cod  ./setup.sh"

eval "${COMMAND}"
