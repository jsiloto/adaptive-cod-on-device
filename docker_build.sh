#!/bin/bash

docker build -t juliano.siloto/adaptive_cod_on_device \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USER=$(id -un) \
  --build-arg GROUP=$(id -gn) .