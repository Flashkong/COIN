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


from detectron2.config import CfgNode as CN


def add_config(cfg):
    """
    Add config.
    """
    _C = cfg

    _C.RESUME = False
    # ---------------------------------------------------------------------------- #
    # SOLVER Settings
    # ---------------------------------------------------------------------------- #
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 3
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.REFERENCE_WORLD_SIZE = 0
    _C.SOLVER.PER_MODULE_PARAM_WEIGHT = []

    # ---------------------------------------------------------------------------- #
    # DATASETS Settings
    # ---------------------------------------------------------------------------- #
    _C.DATASETS.TRAIN_UNLABEL = ("",)
    _C.DATASETS.STYLE_NAME = ""

    # ---------------------------------------------------------------------------- #
    # TEST Settings
    # ---------------------------------------------------------------------------- #

    _C.TEST.EVALUATOR = "VOCeval"
    _C.TEST.DETECTIONS_PER_IMAGE = 100
    _C.TEST.SAVE_DETECTION_PKLS = False

    # ---------------------------------------------------------------------------- #
    # Input Settings
    # ---------------------------------------------------------------------------- #
    _C.INPUT.TEACHER_CLOUD = CN()
    _C.INPUT.TEACHER_CLOUD.MIN_SIZE_TEST = 600
    _C.INPUT.TEACHER_CLOUD.MAX_SIZE_TEST = 1333
    _C.INPUT.TEACHER_CLOUD.FORMAT = "RGB"
    _C.INPUT.TEACHER_CLOUD.NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # grounding dino
    # for collecting results
    _C.INPUT.TEACHER_CLOUD.COLLECT_AUG = ""  # options: "" "ZOOM&AUG" "ZOOM" "AUG"
    _C.INPUT.TEACHER_CLOUD.MIN_CENTER_ZOOM_SIZE = 320
    # for backbone
    _C.INPUT.TEACHER_OFFLINE = CN()
    _C.INPUT.TEACHER_OFFLINE.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073] # CLIP
    _C.INPUT.TEACHER_OFFLINE.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

    _C.INPUT.MIN_SIZE_TRAIN = (600,)
    _C.INPUT.MIN_SIZE_TEST = 600

    # ---------------------------------------------------------------------------- #
    # RESNET Settings
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RESNETS.DEPTH = 50
    _C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone
    _C.MODEL.RESNETS.NORM = "FrozenBN"

    # ---------------------------------------------------------------------------- #
    # ROI_HEADS Settings
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_HEADS.TEACHER_OFFLINE = "CLIPRes5ROIHeads"

    # ---------------------------------------------------------------------------- #
    # Other Model Settings
    # ---------------------------------------------------------------------------- #
    _C.MODEL.TEACHER_CLOUD = CN()
    _C.MODEL.TEACHER_CLOUD.META_ARCHITECTURE = ''
    _C.MODEL.TEACHER_CLOUD.USE_DINO_TYPE_FILTER = False
    _C.MODEL.TEACHER_CLOUD.PROCESSOR_ARCHITECTURE = ''
    _C.MODEL.TEACHER_CLOUD.COLLECT_ARCHITECTURE = ''
    _C.MODEL.TEACHER_CLOUD.TYPE = ''
    _C.MODEL.TEACHER_CLOUD.CONFIG_PATH = ''
    _C.MODEL.TEACHER_CLOUD.WEIGHT = ''
    _C.MODEL.TEACHER_CLOUD.TEST_THRESHOLD = 0.25
    _C.MODEL.TEACHER_CLOUD.PER_CLASS_TEST = False
    _C.MODEL.TEACHER_CLOUD.TOKEN = ''  # for Grounding DINO 1.5 API
    _C.MODEL.TEACHER_OFFLINE = CN()
    _C.MODEL.TEACHER_OFFLINE.META_ARCHITECTURE = 'CLIP'
    _C.MODEL.TEACHER_OFFLINE.COLLECT_ARCHITECTURE = 'CLIP_COLLECTOR'
    _C.MODEL.TEACHER_OFFLINE.TYPE = ''
    _C.MODEL.TEACHER_OFFLINE.TEXT_ENCODER = 'CLIP_TEXT'
    _C.MODEL.ROI_HEADS.POOLING_TYPE = 'meanpool' # attnpool or meanpool
    _C.MODEL.MERGE = 'CKGNet'
    _C.MODEL.MERGE_DIM = 1024

    _C.MODEL.REGION_CLIP = False # CLIP's parameters are used in our paper.


    # ---------------------------------------------------------------------------- #
    # COIN Settings
    # ---------------------------------------------------------------------------- #
    _C.CLOUD = CN()
    _C.CLOUD.Trainer = ""
    _C.CLOUD.PRE_TRAIN_NAME = ''
    _C.CLOUD.BURN_UP_STEP = 45000
    _C.CLOUD.PROTOTYPE_UPDATE_START = 5000  # -1 means do not update
    _C.CLOUD.OFFLINE_TEACHER_UPDATE_ITER = 1
    _C.CLOUD.EMA_KEEP_RATE_OFFLINE = 0.9996
    _C.CLOUD.UPDATE_BACKBONE = False
    _C.CLOUD.ADD_PROMPT_NUM = 4
    _C.CLOUD.CLS_B_THRESH = 0.7
    _C.CLOUD.PROTOTYPE_UPDATE_WEIGHT = 0.9996
    
    _C.CLOUD.NMS_METHOD = 'ms'

    # loss type
    _C.CLOUD.LOSS_TYPE = 'MILCrossEntropy'
    _C.CLOUD.BG_TRAIN = True

    # weights
    _C.CLOUD.CLASSES_WEIGHT = []
    _C.CLOUD.LOSS_BOX_REG_WEIGHT = 1.0
    _C.CLOUD.LOSS_BOX_REG_OFFLINE_WEIGHT = 1.0
    _C.CLOUD.LOSS_BOX_REG_ONLINE_WEIGHT = 1.0
    _C.CLOUD.LOSS_CLS_WEIGHT = 1.0
    _C.CLOUD.LOSS_TEXT_ALIGN_WEIGHT = 10.0
    _C.CLOUD.LOSS_CLS_B_WEIGHT = 0.1
    _C.CLOUD.LOSS_DISTILLATION_WEIGHT = 0.1

    # collect setting
    _C.CLOUD.TEACHER_CLOUD = CN()
    _C.CLOUD.TEACHER_CLOUD.RPN_SEPARATE_COLLECT = False  # if set False, then RCNN_THRESH is set the same as RPN_THRESH.
    _C.CLOUD.TEACHER_CLOUD.RPN_THRESH = 0.25
    _C.CLOUD.TEACHER_CLOUD.RCNN_THRESH = 0.25
    _C.CLOUD.TEACHER_CLOUD.ZOOM_MATCHER_THRESH = 0.6
    _C.CLOUD.TEACHER_CLOUD.COLLECT_NMS_THRESH = 0.6

    _C.CLOUD.MATCHER = CN()
    _C.CLOUD.MATCHER.IOU_THRESHOLDS = 0.5
