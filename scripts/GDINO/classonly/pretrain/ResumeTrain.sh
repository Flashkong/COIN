

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/CLASSONLY/CLIPDET_foggy.yaml\
     MODEL.WEIGHTS  your_checkpoint\
     OUTPUT_DIR output_GDINO_classonly/foggy/pretrain/CLIP_resume