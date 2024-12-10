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

import copy
import logging
import numpy as np
from PIL import Image
import torch

from coin.data.transforms.augmentation_impl import GDINOResize,GDINOZOOM
from coin.data.transforms.transform import NormalizeTransform,WeakAUGTransform
from coin.data.detection_utils import build_strong_augmentation
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from fvcore.transforms.transform import HFlipTransform,VFlipTransform
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.config import configurable

def check_image_size(logger, dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            logger.info(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation. " \
                + f"Use  {expected_wh} for anno file as gt."
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]

class TESTMapper(DatasetMapper):

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            "cfg": cfg
        })
        return ret

    @configurable
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        check_image_size(self.logger, dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class GDINOMapper(DatasetMapper):
    def __init__(self, cfg):
        is_train=False
        self.augmentations = T.AugmentationList(build_GDINO_resize(cfg))
        
        self.compute_tight_boxes = False

        # fmt: off
        self.img_format = cfg.INPUT.TEACHER_CLOUD.FORMAT
        self.mask_on = False
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = False
        self.load_proposals = False
        self.keypoint_hflip_indices = None

        self.is_train = is_train
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())

    def __call__(self, dataset_dict):
        """
        test only
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # dataset_dict["img_pil"] = Image.fromarray(image.astype("uint8"), "RGB")
        check_image_size(self.logger, dataset_dict, image)
        
        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))

        assert self.is_train==False
        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict
    

class COLLECTMapper(DatasetMapper):
    def __init__(self, cfg):
        is_train=False
        self.resize = T.AugmentationList(build_GDINO_resize(cfg,Norm=False))
        self.norm = T.AugmentationList([build_GDINO_norm(cfg)])
        if cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG=="ZOOM&AUG":
            self.zoom = T.AugmentationList(build_GDINO_zoom(cfg))
            self.aug = T.AugmentationList(build_GDINO_aug(cfg))
        elif cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG=="AUG":
            self.aug = T.AugmentationList(build_GDINO_aug(cfg))
            pass
        elif cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG=="ZOOM":
            self.zoom = T.AugmentationList(build_GDINO_zoom(cfg))
        elif cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG=="":
            pass
        else:
            raise NotImplementedError
        
        self.compute_tight_boxes = False

        # fmt: off
        self.img_format = cfg.INPUT.TEACHER_CLOUD.FORMAT
        self.mask_on = False
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = False
        self.load_proposals = False
        self.keypoint_hflip_indices = None

        self.is_train = is_train
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())

    def __call__(self, dataset_dict):
        """
        test only
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        check_image_size(self.logger, dataset_dict, image)
        dataset_dict["img_pil"] = Image.fromarray(image.astype("uint8"), "RGB")
        
        aug_input = T.AugInput(image, sem_seg=None)
        self.resize(aug_input)
        image_resize = aug_input.image
        if hasattr(self,"aug"):
            temp = T.AugInput(image_resize, sem_seg=None)
            self.aug(temp)
            image_aug = temp.image
            dataset_dict["image_aug"] = torch.as_tensor(np.ascontiguousarray(image_aug))
        if hasattr(self,"zoom"):
            temp = T.AugInput(image_resize, sem_seg=None)
            transforms = self.zoom(temp)
            image_zoom = temp.image # 这里计算坐标是相对于原始图片的
            info_dict = get_GDINO_zoom_info(dataset_dict['width'], dataset_dict['height'], image_resize.shape[1], transforms[0].crop_w, transforms[0].crop_h)
            dataset_dict["image_zoom"] = torch.as_tensor(np.ascontiguousarray(image_zoom))
            dataset_dict["zoom_info"] = info_dict
        self.norm(aug_input)
        image_ori = aug_input.image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_ori))
        assert self.is_train==False
        # USER: Modify this if you want to keep them for some reason.
        # dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict


def build_GDINO_resize(cfg,Norm=True):
    min_size = cfg.INPUT.TEACHER_CLOUD.MIN_SIZE_TEST
    max_size = cfg.INPUT.TEACHER_CLOUD.MAX_SIZE_TEST 
    augmentation = [GDINOResize(min_size,max_size)]
    if Norm:
        augmentation.append(build_GDINO_norm(cfg))
    return augmentation


def build_GDINO_norm(cfg):
    norm = cfg.INPUT.TEACHER_CLOUD.NORM
    return NormalizeTransform(norm[0],norm[1])

def build_GDINO_zoom(cfg):
    crop_size = cfg.INPUT.TEACHER_CLOUD.MIN_CENTER_ZOOM_SIZE
    min_size = cfg.INPUT.TEACHER_CLOUD.MIN_SIZE_TEST
    max_size = cfg.INPUT.TEACHER_CLOUD.MAX_SIZE_TEST 
    return [GDINOZOOM(crop_size),GDINOResize(min_size,max_size),build_GDINO_norm(cfg)]

def build_GDINO_aug(cfg):
    return [WeakAUGTransform(),build_GDINO_norm(cfg)]

def get_GDINO_zoom_info(original_w, original_h, resize_w, crop_w, crop_h):
    zoom_ratio = original_w / resize_w
    ratio = crop_w / crop_h
    crop_h = int(round(crop_h * zoom_ratio,0))
    crop_w = int(round(crop_h * ratio,0))
    x1 = int(round((original_w - crop_w) / 2,0))
    y1 = int(round(x1/ratio,0))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    info_dict = {"leftup":(x1,y1),"rightdown":(x2,y2),"crop_size":(crop_w,crop_h)}
    return info_dict


class DatasetMapperUnsupervised(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)
        
        if cfg.INPUT.CROP.ENABLED and is_train: # False by default
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        check_image_size(self.logger, dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=None)
        transforms = aug_input.apply_augmentations(self.augmentation)
        dataset_dict["random_flip"] = "no"
        for trans in transforms:
            if isinstance(trans,HFlipTransform):
                dataset_dict["random_flip"] = "horizontal"
            elif isinstance(trans,VFlipTransform):
                dataset_dict["random_flip"] = "vertical"
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)


class DatasetMapperUnsupervisedWithGT(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        check_image_size(self.logger, dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=None)
        transforms = aug_input.apply_augmentations(self.augmentation)

        dataset_dict["random_flip"] = "no"
        for trans in transforms:
            if isinstance(trans,HFlipTransform):
                dataset_dict["random_flip"] = "horizontal"
            elif isinstance(trans,VFlipTransform):
                dataset_dict["random_flip"] = "vertical"
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            # 对于CROP的情况，重新覆盖框
            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)
