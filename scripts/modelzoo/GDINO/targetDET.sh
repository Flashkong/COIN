python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/foggy.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/foggy/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/foggy/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/cityscape.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/cityscape/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/cityscape/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/BDD100K.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/BDD100K/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/BDD100K/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/clipart.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/clipart/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/clipart/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/KITTI.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/KITTI/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/KITTI/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/SIM.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/SIM/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/SIM/targetDET
