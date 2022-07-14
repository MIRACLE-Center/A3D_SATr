.#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
echo ${@:3}
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    ./deeplesion/train_dist.py $CONFIG --launcher pytorch ${@:3}
