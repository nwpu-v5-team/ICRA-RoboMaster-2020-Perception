import registry
import torch
import numpy

class SeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
        super().__init__()
        ReLU = torch.nn.ReLU if onnx_compatible else torch.nn.ReLU6
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      groups=in_channels, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(in_channels),
            ReLU(),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)

@registry.BoxPredictor.register("YoloLayer")
class YoloLayer(torch.nn.Module):
    def __int__(self):
        super(YoloLayer, self).__int__()
        self.ModuleList = torch.nn.ModuleList()
        pass



@registry.BoxPredictor.register("BoxPredict-SSD")
class BoxPredict(torch.nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.class_num = setting["class_num"]
        out_channels_List, box_num_List = setting["out_channels"], setting["box_num"]
        self.cls_headers = torch.nn.ModuleList()
        self.reg_headers = torch.nn.ModuleList()
        for level, (out_channels, box_num) in enumerate(zip(out_channels_List, box_num_List)):
            if level == len(out_channels_List) -1:
                self.cls_headers.append(
                   torch.nn.Conv2d(out_channels,box_num * self.class_num,kernel_size=1)
                )
                self.reg_headers.append(
                    torch.nn.Conv2d(out_channels, box_num*4, kernel_size=1)
                )

            else :
                self.cls_headers.append(
                    SeparableConv2d(out_channels, box_num*self.class_num,kernel_size=3, stride=1, padding=1)
                )
                self.reg_headers.append(
                    SeparableConv2d(out_channels,box_num*4, kernel_size=3,stride=1,padding=1)
                )

        self.reset_param()
    def reset_param(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    def forward(self, features_list):
        cls_logits = []
        bbox_pred = []

        batch_size = features_list[0].shape[0]
        for feature , cls_Module, reg_Module in zip(features_list, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_Module(feature).permute(0,2,3,1).contiguous().view(batch_size, -1, self.class_num))
            bbox_pred.append(reg_Module(feature).permute(0,2,3,1).contiguous().view(batch_size, -1, 4))
        cls_logits = torch.cat(cls_logits,dim=1)
        bbox_pred = torch.cat(bbox_pred,dim=1)
        return cls_logits, bbox_pred



# TODO SSDLITE


@registry.BoxPredictor.register("BoxPredictLite-SSD")
class BoxPredictLite(torch.nn.Module):
    def __int__(self):
        super(BoxPredictLite, self).__int__()
        self.cls_ModuleList = torch.nn.ModuleList()
        self.reg_ModuleList = torch.nn.ModuleList()
        pass



