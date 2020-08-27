import torch
import registry
from math import sqrt
from itertools import  product

#
#  先验框 anchor-base方法
#  基于锚点方法，可以提高检测精确度
#
@registry.PriorBox.register("PriorBox")
class PriorBox:
    def __init__(self, setting_dict):
        self.image_size = 512
        self.feature_maps = setting_dict["FEATURE_MAPS"]
        self.min_sizes = setting_dict["MIN_SIZES"]
        self.max_sizes = setting_dict["MAX_SIZES"]
        self.strides = setting_dict["STRIDES"]
        self.aspect_ratios = setting_dict["ASPIECT_RATIOS"]
        self.clip = setting_dict["CLIP"]
    def __call__(self):

        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale  ## why
                cy = (i + 0.5) / scale  ## why

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h * ratio])
                    priors.append([cx, cy, w / ratio, h / ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
