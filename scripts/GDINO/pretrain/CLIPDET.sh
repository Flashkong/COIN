#!/bin/bash

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_foggy.yaml \
     OUTPUT_DIR output_GDINO/foggy/pretrain/CLIPDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_cityscape.yaml\
     OUTPUT_DIR output_GDINO/cityscape/pretrain/CLIPDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_BDD100K.yaml\
     OUTPUT_DIR output_GDINO/BDD100K/pretrain/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_KITTI.yaml\
     OUTPUT_DIR output_GDINO/KITTI/pretrain/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_SIM.yaml\
     OUTPUT_DIR output_GDINO/SIM/pretrain/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_clipart.yaml\
     OUTPUT_DIR output_GDINO/clipart/pretrain/CLIPDET