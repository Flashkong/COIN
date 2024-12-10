#!/bin/bash

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/foggy.yaml \
     MODEL.TEACHER_CLOUD.META_ARCHITECTURE GDINO1_5_API \
     MODEL.TEACHER_CLOUD.PROCESSOR_ARCHITECTURE GDINO_1_5_API_PROCESSOR \
     MODEL.TEACHER_CLOUD.COLLECT_ARCHITECTURE GDINO_COLLECTOR \
     MODEL.TEACHER_CLOUD.TOKEN your_token \
     MODEL.WEIGHTS your_pretrain_model+your_online_collect_results \
     OUTPUT_DIR output_GDINO1.5API/foggy/gard/targetDet
