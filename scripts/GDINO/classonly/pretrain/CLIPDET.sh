#!/bin/bash

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/CLASSONLY/CLIPDET_foggy.yaml \
     OUTPUT_DIR output_GDINO_classonly/foggy/pretrain/CLIP
