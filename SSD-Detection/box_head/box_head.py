import torch
import numpy

from box_head.box_predicion import build_boxpredict
from box_head.box_loss import  build_boxlossEval
from anchors.anchor_base import build_priorbox
from data.transform import TargetTransform
from box_head.box_nms.process import Processor, processor2
from utils import  box_utils

import registry


@registry.BoxHead.register("SSD-boxhead")
class BoxHead(torch.nn.Module):
    def __init__(self, setting_dict):
        super().__init__()
        self.predictor = build_boxpredict(setting_dict["predictor"])
        self.loss_eval = build_boxlossEval(setting_dict["boxloss"])
        self.processor = Processor()
        self.center_var= setting_dict["center_var"]
        self.size_var  = setting_dict["size_var"]
        self.priors    = build_priorbox(setting_dict["prior_box"])()
        self.target_Transform = TargetTransform(self.priors,self.center_var, self.size_var, 0.5)

    def _train(self, cls_logits, bbox_pred, targets):

        result = [(self.target_Transform(boxes, index))for boxes,index in zip(targets["boxes"], targets["labels"])]
        groudTruth_boxes, groundTruth_label = [res[0][None,...] for res in result],[res[1][None,...] for res in result]
        groundTruth_label = torch.as_tensor(torch.cat(groundTruth_label,dim=0),dtype=torch.long)
        groudTruth_boxes = torch.cat(groudTruth_boxes, dim=0)


        reg_loss, cls_loss = self.loss_eval(cls_logits, bbox_pred, groudTruth_boxes.to(cls_logits.device), groundTruth_label.to(cls_logits.device))

        loss_dict =dict(
            reg_loss = reg_loss,
            cls_loss = cls_loss
        )
        detection  = (cls_logits, bbox_pred)
        return detection, loss_dict

    def _test(self, cls_logits, bbox_pred):


        boxes = box_utils.location2box(bbox_pred, self.priors.to(bbox_pred.device),
                                                    self.center_var,
                                                    self.size_var)
        scores = torch.nn.functional.softmax(cls_logits, dim=2)

        boxes = box_utils.center2corner(boxes)

        detections = (scores, boxes)
        detections = self.processor(detections)
        return detections, {}

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)

        if self.training :
            return self._train(cls_logits, bbox_pred, targets)
        else :
            return self._test(cls_logits, bbox_pred)

