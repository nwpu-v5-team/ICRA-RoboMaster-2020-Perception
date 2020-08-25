import numpy as np
import torch

from utils.box_utils import *




# taregtTransform

class TargetTransform:
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = center2corner(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)

        boxes = corner2center(boxes)
        locations = box2location(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)

        return locations, labels

