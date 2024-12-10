python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP-GLIP_foggy.yaml \
     OUTPUT_DIR output_GLIP/foggy/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP-GLIP_cityscape.yaml \
     OUTPUT_DIR output_GLIP/cityscape/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP-GLIP_BDD100K.yaml \
     OUTPUT_DIR output_GLIP/BDD100K/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP-GLIP_KITTI.yaml \
     OUTPUT_DIR output_GLIP/KITTI/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP-GLIP_SIM.yaml \
     OUTPUT_DIR output_GLIP/SIM/test_CLIP
