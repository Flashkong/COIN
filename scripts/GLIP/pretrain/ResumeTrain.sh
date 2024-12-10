#!/bin/bash

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_foggy.yaml \
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GLIP/foggy/pretrain/CLIPDET_resume

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_cityscape.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GLIP/cityscape/pretrain/CLIPDET_resume

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_BDD100K.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GLIP/BDD100K/pretrain/CLIPDET_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_KITTI.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GLIP/KITTI/pretrain/CLIPDET_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIP-GLIP-DET_SIM.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GLIP/SIM/pretrain/CLIPDET_resume
