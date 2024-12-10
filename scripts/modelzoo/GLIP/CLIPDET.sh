python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/foggy.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/foggy/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/foggy/CLIPDET

python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/cityscape.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/cityscape/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/cityscape/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/BDD100K.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/BDD100K/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/BDD100K/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/KITTI.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/KITTI/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/KITTI/CLIPDET


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/SIM.yaml \
     MODEL.WEIGHTS model_zoo/GLIP/SIM/CLIPDET.pth \
     CLOUD.Trainer ModelZoo_test \
     OUTPUT_DIR output_modelzoo/GLIP/SIM/CLIPDET
