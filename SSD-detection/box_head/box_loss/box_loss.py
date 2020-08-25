import torch
import registry

from utils import box_utils



#
#  ssd-loss-function
#
@registry.BoxLossEval.register("MutiBoxLoss")
class MutiBoxLoss(torch.nn.Module):
    def __init__(self, setting_dict):
        super(MutiBoxLoss, self).__init__()
        self.neg_pos_ratio = setting_dict["ratio"]
        self.class_num = setting_dict["class_num"]
    def forward(self, pred, bbox, gt_bbox, label):
        label.to(pred.device)
        gt_bbox.to(pred.device)
        with torch.no_grad():
            loss = -torch.nn.functional.log_softmax(pred, dim=2)[:,:,0]
            mask = box_utils.negative_supress(loss, label, self.neg_pos_ratio)
        pred = pred[mask,:]
        pred_loss = torch.nn.functional.cross_entropy(pred.view(-1,self.class_num), label[mask], reduction="sum")

        pos_mask = label > 0
        bbox = bbox[pos_mask,:].view(-1,4)
        gt_bbox = gt_bbox[pos_mask,:].view(-1, 4)
        L1_loss = torch.nn.functional.smooth_l1_loss(bbox, gt_bbox, reduction="sum")
        num_pos = gt_bbox.size(0)
        return L1_loss/num_pos , pred_loss/num_pos
