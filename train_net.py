# Copyright (c) Shuaifeng Li at UESTC. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import launch
# to register
import coin.data.datasets.builtin
import coin.modeling.meta_arch
import coin.modeling.text_encoder
import coin.modeling.backbone
import coin.modeling.merge
import coin.modeling.roi_heads
import coin.modeling.proposal_generator
from coin.layers.nms import mynms
from coin.utils.util import copy_codes, default_setup, default_argument_parser
from coin import add_config
from coin.engine import GDINOTrainer,CoinTrainer,CLIPTrainer,PRETrainer,CoinTrainer,OracleTrainer,GLIPTrainer
from coin.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from shutil import copyfile
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # detectron2的配置文件
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['RESUME',args.resume])
    if cfg.CLOUD.Trainer == "GDINO" or cfg.CLOUD.Trainer == "GLIP" or cfg.CLOUD.Trainer == "CLIP" or cfg.CLOUD.Trainer == 'ModelZoo_test':
        args.eval_only=True
    cfg.freeze()
    # 设置logger之类的，输出环境信息等
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    with open(os.path.join(cfg.OUTPUT_DIR, 'note.txt'),'a',encoding='utf-8') as f:
        f.write(args.info)
    copyfile(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'))
    for i in PathManager.open(args.config_file, "r", encoding="utf-8").readlines():
        if '_BASE_'in i :
            base_file = i.split('"')[1]
            copyfile(os.path.join('/'.join(args.config_file.split('/')[:-1]), base_file), os.path.join(cfg.OUTPUT_DIR, 'base.yaml'))
            break
    copy_codes('coin', os.path.join(cfg.OUTPUT_DIR, 'coin'), recursion=True)

    if cfg.CLOUD.Trainer == "GDINO":
        Trainer = GDINOTrainer
    elif cfg.CLOUD.Trainer == "GLIP":
        Trainer = GLIPTrainer
    elif cfg.CLOUD.Trainer == "CLIP":
        Trainer = CLIPTrainer
    elif cfg.CLOUD.Trainer == "PRETRAIN":
        Trainer = PRETrainer
    elif cfg.CLOUD.Trainer == "CoinTrainer":
        Trainer = CoinTrainer
    elif cfg.CLOUD.Trainer=="ORACLE":
        Trainer = OracleTrainer
    elif cfg.CLOUD.Trainer=="ModelZoo_test": # only for testing model zoo
        Trainer = CoinTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    mynms.update(cfg.CLOUD.NMS_METHOD)

    if args.eval_only:
        if cfg.CLOUD.Trainer in ["CoinTrainer"]:
            trainer = Trainer(cfg)
            trainer.resume_or_load(resume=cfg.RESUME)
            # 非常需要注意！！！！ 他们也是用student来测试的
            if args.test_model_role == 'targetdet':
                res = Trainer.test(cfg, trainer.model)
            elif args.test_model_role == 'clipdet':
                res = Trainer.test(cfg, trainer.offline_teacher)
            else:
                raise NotImplementedError
        elif cfg.CLOUD.Trainer in ["GDINO", "GLIP"]:
            model = Trainer.build_model(cfg)
            model = Trainer.load_model(cfg,model)
            res = Trainer.test(cfg, model)
        elif cfg.CLOUD.Trainer in ["CLIP"]:
            model_CLOUD, model_CLIP = Trainer.build_model(cfg)
            model_CLOUD = Trainer.load_model(cfg,model_CLOUD)
            Trainer.collect_results(model_CLOUD, model_CLIP)
            res = Trainer.test(cfg, model_CLOUD)
            res = Trainer.test(cfg, model_CLIP)
        elif cfg.CLOUD.Trainer in ["ModelZoo_test"]:
            model = Trainer.build_model(cfg)
            model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    if isinstance(trainer, PRETrainer):
        if cfg.MODEL.WEIGHTS=='':
            trainer.collect_results()
        else:
            trainer.delete_model()
            trainer.resume_or_load(resume=cfg.RESUME)
        trainer.train()
    else:
        trainer.resume_or_load(resume=cfg.RESUME)
        return trainer.train()

if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()

    # launch文件启动，主要进行分布式训练
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url='auto',
        args=(args,),
    )
