import registry
from backbone.mobileNet.mobilenetV2 import mobilenet_v2
from backbone.mobileNet.mobilenetV3 import MobileNetV3_Large
from backbone.wangshuaiNet.wsNet import getWS
from backbone.VGG.vgg import VGG

def build_backbone(backbone_name):
    print(backbone_name)
    return registry.BackBone[backbone_name]()
