from detectron2.data.transforms import Augmentation
import random
from PIL import ImageFilter
from PIL import ImageOps

from coin.data.transforms.transform import ResizeTransform,CenterCropTransform

class GDINOResize(Augmentation):
    def __init__(self, min_size, max_size=None):
        assert isinstance(min_size, int)
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, self.max_size)

        img_size = (image.shape[1],image.shape[0])
        size = get_size(img_size, self.min_size, self.max_size)

        return ResizeTransform(img_size,size)
    
class GDINOZOOM(Augmentation):
    def __init__(self, min_size):
        assert isinstance(min_size, int)
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w,h = image.shape[1],image.shape[0] # w,h
        ratio = w/h
        if ratio >= 1: 
            crop_w = self.min_size * ratio
            crop_h = self.min_size
        elif ratio < 1:
            crop_w = self.min_size
            crop_h = self.min_size / ratio
        return CenterCropTransform(int(round(crop_w,0)),int(round(crop_h,0)))
    

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
