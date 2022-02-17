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


COMMAND="docker run --gpus device=0 --rm -it \
            --shm-size=32G  \
            -v /work/juliano.siloto/datasets:/work/resource/dataset  \
            -v $PWD:/work -w /work/server \
            -u $(id -u):$(id -g)  \
            --cap-add LINUX_IMMUTABLE \
            --userns=host  \
            -p 5000:5000 \
            -e NVIDIA_VISIBLE_DEVICES=\"0\" \
            -e WANDB_API_KEY=7db49f6a22d959fcba5d6bbaee1f7c28732dd745 \
            --name juliano.siloto.acod-server.gpu0 \
            juliano.siloto/adaptive_cod ./setup.sh"

eval "${COMMAND}"
