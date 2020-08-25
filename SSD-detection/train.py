import argparse
import torch
import time
import logging
from solver.trainer import do_train_one_epoch, do_evaluate
from solver import build_LRscheduler
from solver import build_optimizer
from data.dataset import make_dataLoader
from utils.checkpoint import CheckPoint
from ssd_detector import *

import utils.setting_dict

def train(setting_dict):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SSD.trainer")
    logger.setLevel(logging.INFO)
    logger.info("start training....")

    model = SSDdetector(setting_dict=setting_dict["model"])

    ## if you want to fine tune the pretrained model
    ## just change "the path of pretrained model" to your model
    if setting_dict["fine_tune"] :
        checkpoint = torch.load(setting_dict["predtrained_model"], map_location=torch.device("cpu"))
        model_dict = {}
        for key, value in checkpoint.pop("model").items():
            if "backbone" in key:
                model_dict[key.replace("backbone.","")] = value
        model.backbone.load_state_dict(model_dict)
        model.load_state_dict(checkpoint.pop("model"))

        for para in model.backbone.parameters() :
            para.requires_grad = False


    device = torch.device(setting_dict["device"])
    model.to(device)
    lr = setting_dict["solver"]["LR"]


    ## if you want to fine tune the pretrained model
    ## just change model to model.boxhead
    optimizer = build_optimizer(setting_dict["solver"]["optimizer"], model, lr)
    scheduler = build_LRscheduler(setting_dict["solver"]["LRscheduler"])(optimizer,
                                                                         setting_dict["solver"]["LR_STEP"])
    train_loader = make_dataLoader(setting_dict["train"], True)
    test_loader = make_dataLoader(setting_dict["test"], False)
    checkpointer = CheckPoint(model, optimizer, scheduler, "", logger)
    print(setting_dict["train_epoch"])
    for i in range(1,setting_dict["train_epoch"] +1):
        do_train_one_epoch(model,train_loader,optimizer,scheduler,device,setting_dict["out_dir"], i)
        if i % 1 == 0 :
            do_evaluate(model, test_loader, device,setting_dict["out_dir"], i)
        if i % 7 == 0 :
            checkpointer.save(setting_dict["out_dir"]+"/v3_model_{:06d}".format(i))
    checkpointer.save("finial")
    return model


def main():
    torch.backends.cudnn.benchmark = True

    parse = argparse.ArgumentParser(description="PYtorch-SSD train process")

    parse.add_argument("--out_dir", type=str, default=".")
    parse.add_argument("--fine_tune", type=bool, default=False)
    parse.add_argument("--pretrained_model", type=str, default="")

    args = parse.parse_args()
    utils.setting_dict.setting_dict["out_dir"] = args.out_dir + utils.setting_dict.setting_dict["out_dir"]
    utils.setting_dict.setting_dict["fine_tune"] = args.fine_tune
    utils.setting_dict.setting_dict["pretrained_model"] = args.pretrained_model

    model = train(setting_dict=utils.setting_dict.setting_dict)
    torch.cuda.empty_cache()

if __name__ == "__main__" :
  main()

