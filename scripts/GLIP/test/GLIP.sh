python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GLIP_foggy.yaml \
     OUTPUT_DIR output_GLIP/foggy/test_GLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GLIP_cityscape.yaml \
     OUTPUT_DIR output_GLIP/cityscape/test_GLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GLIP_BDD100K.yaml \
     OUTPUT_DIR output_GLIP/BDD100K/test_GLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GLIP_KITTI.yaml \
     OUTPUT_DIR output_GLIP/KITTI/test_GLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GLIP_SIM.yaml \
     OUTPUT_DIR output_GLIP/SIM/test_GLIP
