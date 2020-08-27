
import registry
from box_head.box_loss.box_loss import  MutiBoxLoss

def build_boxlossEval(loss):
    loss_name = loss["name"]
    setting = loss["setting"]
    return registry.BoxLossEval[loss_name](setting)



#a = build_boxlossEval("MutiBoxLoss", {"ratio" : 1, "class_num" : 10})