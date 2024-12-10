#!/bin/bash
python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_foggy.yaml \
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO/foggy/pretrain/CLIPDET_resume

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_cityscape.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO/cityscape/pretrain/CLIPDET_resume

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_BDD100K.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO/BDD100K/pretrain/CLIPDET_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_KITTI.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO/KITTI/pretrain/CLIPDET_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_SIM.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO/SIM/pretrain/CLIPDET_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/PRETRAINS/CLIPDET_clipart.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO/clipart/pretrain/CLIPDET_resume