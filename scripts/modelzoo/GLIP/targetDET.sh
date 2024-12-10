python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/foggy.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/foggy/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/foggy/targetDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/cityscape.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/cityscape/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/cityscape/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/BDD100K.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/BDD100K/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/BDD100K/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/KITTI.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/KITTI/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/KITTI/targetDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/SIM.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/SIM/targetDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/SIM/targetDET
