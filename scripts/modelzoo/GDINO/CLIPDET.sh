python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/foggy.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/foggy/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/foggy/CLIPDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/cityscape.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/cityscape/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/cityscape/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/BDD100K.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/BDD100K/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/BDD100K/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/clipart.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/clipart/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/clipart/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/KITTI.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/KITTI/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/KITTI/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GDINO/SIM.yaml \
     MODEL.WEIGHTS model_zoo/GDINO/SIM/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GDINO/SIM/CLIPDET
