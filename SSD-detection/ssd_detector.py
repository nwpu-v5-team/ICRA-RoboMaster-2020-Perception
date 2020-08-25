import torch

from backbone import build_backbone
from box_head import build_boxhead

class SSDdetector(torch.nn.Module):
    def __init__(self, setting_dict):
        super(SSDdetector, self).__init__()
        self.setting_dict = setting_dict
        self.backbone = build_backbone(self.setting_dict["backbone"])
        self.box_head = build_boxhead(self.setting_dict["boxhead"])

    def forward(self, images, targets=None):
        features = self.backbone(images)
        #torch.onnx.export(self.backbone, images, "model.onnx", opset_version=9, export_params=True)
        detection, detection_loss = self.box_head(features, targets)

        if self.training :
            return  detection_loss
        return detection
