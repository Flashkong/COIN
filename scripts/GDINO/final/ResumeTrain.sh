python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/foggy.yaml \
     MODEL.WEIGHTS your_checkpoint \
     OUTPUT_DIR output_GDINO/foggy/gard/targetDet_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/cityscape.yaml \
     MODEL.WEIGHTS your_checkpoint \
     OUTPUT_DIR output_GDINO/cityscape/gard/targetDet_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/BDD100K.yaml \
     MODEL.WEIGHTS your_checkpoint \
     OUTPUT_DIR output_GDINO/BDD100K/gard/targetDet_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/KITTI.yaml \
     MODEL.WEIGHTS your_checkpoint \
     OUTPUT_DIR output_GDINO/KITTI/gard/targetDet_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/SIM.yaml \
     MODEL.WEIGHTS your_checkpoint \
     OUTPUT_DIR output_GDINO/SIM/gard/targetDet_resume


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/clipart.yaml \
     MODEL.WEIGHTS your_checkpoint \
     OUTPUT_DIR output_GDINO/clipart/gard/targetDet_resume