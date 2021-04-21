#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

root=$2
date=$3
model=$4
resume=""$root"/"$date"/model/"$model""

python -utt example.py --imgpath="" --resume=$resume
