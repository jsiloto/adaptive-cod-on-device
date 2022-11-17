#!/bin/bash

OLDDIR=$(pwd)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/..
python3 benchmark/build_models.py
sleep 5
python3 benchmark/switching_benchmark.py --name "assine2022b" --cpus 4
sleep 5
python3 benchmark/switching_benchmark.py --name "lee2021" --cpus 4
sleep 5
python3 benchmark/switching_benchmark.py --name "matsubara2022" --cpus 4
sleep 5
python3 benchmark/switching_benchmark.py --name "assine2022a" --cpus 4

cd $OLDDIR
