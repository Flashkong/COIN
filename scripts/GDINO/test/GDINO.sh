python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GDINO_foggy.yaml \
     OUTPUT_DIR output_GDINO/foggy/test_GDINO


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GDINO_cityscape.yaml \
     OUTPUT_DIR output_GDINO/cityscape/test_GDINO


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GDINO_BDD100K.yaml \
     OUTPUT_DIR output_GDINO/BDD100K/test_GDINO


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GDINO_KITTI.yaml \
     OUTPUT_DIR output_GDINO/KITTI/test_GDINO


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GDINO_SIM.yaml \
     OUTPUT_DIR output_GDINO/SIM/test_GDINO


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/BASELINES/GDINO_clipart.yaml \
     OUTPUT_DIR output_GDINO/clipart/test_GDINO
