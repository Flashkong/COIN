python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/foggy.yaml \
     MODEL.WEIGHTS your_pretrain_model+your_online_collect_results \
     OUTPUT_DIR output_GLIP/foggy/gard/targetDet


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/cityscape.yaml \
     MODEL.WEIGHTS your_pretrain_model+your_online_collect_results \
     OUTPUT_DIR output_GLIP/cityscape/gard/targetDet


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/BDD100K.yaml \
     MODEL.WEIGHTS your_pretrain_model+your_online_collect_results \
     OUTPUT_DIR output_GLIP/BDD100K/gard/targetDet


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/KITTI.yaml \
     MODEL.WEIGHTS your_pretrain_model+your_online_collect_results \
     OUTPUT_DIR output_GLIP/KITTI/gard/targetDet


python train_net.py \
     --num-gpus 1 \
     --config configs/coin/GLIP/SIM.yaml \
     MODEL.WEIGHTS your_pretrain_model+your_online_collect_results \
     OUTPUT_DIR output_GLIP/SIM/gard/targetDet
