#!/bin/bash
docker run --rm -it --network host  -v $PWD:/work -w /work jupyter/datascience-notebook jupyter notebook --no-browser -NotebookApp.token='ABC'
