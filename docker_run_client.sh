#!/bin/bash

COMMAND="docker run --rm -it \
            --shm-size=32G  \
            -v /work/juliano.siloto/datasets:/work/resource/dataset  \
            -v $PWD:/work -w /work \
            -u $(id -u):$(id -g)  \
            --cap-add LINUX_IMMUTABLE \
            --userns=host  \
            --network host \
            --name juliano.siloto.acod-client  \
            juliano.siloto/adaptive_cod"

eval "${COMMAND}"
