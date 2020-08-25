from .transform import *
from .target_transform import *
from anchors.anchor_base.prior_box import *

def build_transforms(setting_dict,is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(setting_dict["PIXEL_MEAN"]),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(setting_dict["IMAGE_SIZE"]),
            SubtractMeans(setting_dict["PIXEL_MEAN"]),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(setting_dict["IMAGE_SIZE"]),
            SubtractMeans(setting_dict["PIXEL_MEAN"]),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform

