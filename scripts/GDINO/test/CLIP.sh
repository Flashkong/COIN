python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_foggy.yaml \
     OUTPUT_DIR output_GDINO/foggy/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_cityscape.yaml \
     OUTPUT_DIR output_GDINO/cityscape/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_BDD100K.yaml \
     OUTPUT_DIR output_GDINO/BDD100K/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_KITTI.yaml \
     OUTPUT_DIR output_GDINO/KITTI/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_SIM.yaml \
     OUTPUT_DIR output_GDINO/SIM/test_CLIP


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/CLIP_clipart.yaml \
     OUTPUT_DIR output_GDINO/clipart/test_CLIP
