#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-4321}

# usage
if [ $# -ne 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_test.sh [number of gpu] [path to option file]"
    exit
fi


python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    bbcu/test.py -opt $CONFIG --launcher pytorch
