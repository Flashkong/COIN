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

import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from coin.data.datasets.pascal_voc import register_pascal_voc

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        # Cityscapes
        ("citytrain", 'CityScapes_FoggyCityScapes', "train_city", 8, 'png'),
        ("cityval", 'CityScapes_FoggyCityScapes', "val_city", 8, 'png'),
        # Foggy-Cityscapes (all levels). Not used
        ("foggytrain", 'CityScapes_FoggyCityScapes', "train_foggy", 8, 'png'),
        ("foggyval", 'CityScapes_FoggyCityScapes', "val_foggy", 8, 'png'),
        # Foggy-Cityscapes (0.02).  Used
        ("foggytrain_0.02", 'CityScapes_FoggyCityScapes', "train_foggy_0.02", 8, 'png'),
        ("foggyval_0.02", 'CityScapes_FoggyCityScapes', "val_foggy_0.02", 8, 'png'),
        # Cityscapes with one class 'car'. Not used
        ("citytrain_car", 'CityScapes_FoggyCityScapes', "train_city_car", 1, 'png'),
        ("cityval_car", 'CityScapes_FoggyCityScapes', "val_city_car", 1, 'png'),
        # Clipart
        ("cliparttrain", 'clipart', "all", 20, 'jpg'),
        ("clipartval", 'clipart', "all", 20, 'jpg'),
        # KITTI
        ('KITTItrainval','KITTI','train_car',1,'png'),
        # SIM with one class 'car'. Used
        ('SIMtrainval_car','SIM','train_car',1,'jpg'),
        # SIM with all three classes. Not used
        ('SIMtrainval','SIM','train',3,'jpg'),
        # BDD100K
        ('BDD100Ktrain','BDD100K_voc','train_object',7,'jpg'),
        ('BDD100Kval','BDD100K_voc','val_object',7,'jpg'),
    ]
    for name, dirname, split, cls, imgformat in SPLITS:
        year = 2012
        if cls == 20:
            class_names = (
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            )
        elif cls == 8:
            class_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')
            # class_names = ('rider',)
        elif cls==6:
            class_names = ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        elif cls==7:
            class_names = ('person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
        elif cls==3:
            class_names = ('car', 'motorbike', 'person')
        elif cls==1:
            class_names = ('car',)
        else:
            raise RuntimeError
        register_pascal_voc(name, os.path.join(root, dirname), split, year, imgformat, class_names)
        MetadataCatalog.get(name).evaluator_type = "VOCeval"


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_all_pascal_voc(_root)
