#!/bin/bash

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_foggy.yaml \
     OUTPUT_DIR output_GLIP/foggy/pretrain/CLIPDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_cityscape.yaml\
     OUTPUT_DIR output_GLIP/cityscape/pretrain/CLIPDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_BDD100K.yaml\
     OUTPUT_DIR output_GLIP/BDD100K/pretrain/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_KITTI.yaml\
     OUTPUT_DIR output_GLIP/KITTI/pretrain/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_SIM.yaml\
     OUTPUT_DIR output_GLIP/SIM/pretrain/CLIPDET
