from fvcore.transforms.transform import (
    Transform,
)
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from groundingdino.util.misc import interpolate
from PIL import Image,ImageFilter,ImageOps
import random

class ResizeTransform(Transform):
    def __init__(self,source_size, target_size):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        img = Image.fromarray(img.astype("uint8"), "RGB")
        return np.array(F.resize(img, self.target_size))

    def apply_coords(self, coords):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(self.target_size, self.source_size))
        ratio_width, ratio_height = ratios
        scaled_boxes = coords * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        return scaled_boxes
    
class NormalizeTransform(Transform):
    def __init__(self, mean, std):
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        img = F.to_tensor(img)
        return F.normalize(img, mean=self.mean, std=self.std).cpu().numpy()

    def apply_coords(self, coords):
        return coords
    
class CenterCropTransform(Transform):
    def __init__(self, crop_w, crop_h):
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        img = Image.fromarray(img.astype("uint8"), "RGB")
        original_w, original_h = img.size

        x1 = int(round((original_w - self.crop_w) / 2,0))
        y1 = int(round((original_h - self.crop_h) / 2,0))
        x2 = x1 + self.crop_w
        y2 = y1 + self.crop_h
        cropped_img = img.crop((x1, y1, x2, y2))
        return np.array(cropped_img)

    def apply_coords(self, coords):
        return coords
    
class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    def __init__(self, threshold):
        assert 0 < threshold < 1
        self.threshold = round(threshold * 256)

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        attrs = f"(min_scale={self.threshold}"
        return self.__class__.__name__ + attrs
    
class WeakAUGTransform(Transform):
    def __init__(self,):
        super().__init__()
        augmentation = []
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        augmentation.append(transforms.RandomApply([Solarize(threshold=0.5)],
                                                p=0.2))
        self.aug = transforms.Compose(augmentation)
    
    def apply_image(self, img):
        img = Image.fromarray(img.astype("uint8"), "RGB")
        return np.array(self.aug(img))

    def apply_coords(self, coords):
        return coords
