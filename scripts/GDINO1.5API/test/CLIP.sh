python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_foggy.yaml \
     MODEL.TEACHER_CLOUD.PROCESSOR_ARCHITECTURE GDINO1_5_API \
     MODEL.TEACHER_CLOUD.COLLECT_ARCHITECTURE GDINO_COLLECTOR \
     MODEL.TEACHER_CLOUD.TOKEN your_token \
     OUTPUT_DIR output_GDINO1.5API/foggy/test_CLIP