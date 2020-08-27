import registry
import torch
from solver import learning_rate

def build_LRscheduler(learnRate_name):
    return registry.LRScheduler[learnRate_name]

def build_optimizer(setting_dict, model, lr=None):
    # return  torch.optim.AdamW(model.parameters(),lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr,
                           momentum=setting_dict["momentum"],
                           weight_decay=setting_dict["weight_decay"])