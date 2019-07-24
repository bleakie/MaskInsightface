#!/usr/bin/env bash
source /etc/profile
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --pretrained ../models/glink-best/model,1 --network r100 --loss-type 4 --margin-m 0.5 --data-dir ../datasets/glink_aug_datasets_fan/  --prefix ../models/model-aug/model --target 'crop_ours_xiaoyi' 2>&1 > ../models/model-aug/aug.log &

