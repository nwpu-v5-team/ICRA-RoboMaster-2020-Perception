import collections
import datetime
import logging
import os
import time
import torch
from data.dataset import evaluate
from torch.utils.tensorboard import SummaryWriter

def compute(model, data_loader, device):
    result_dict = {}
    model.to(device)
    for batch in data_loader :
        images , (_), indexs = batch
        indexs = indexs.numpy().tolist()
        with torch.no_grad():
            outputs = model(images.to(device))
            result_dict.update(
                {index : result for index, result in zip(indexs, outputs)}
            )
    return result_dict

@torch.no_grad()
def do_evaluate(model, data_loader, device, output_dir, iter=None):
    model.eval()
    result_dict = compute(model, data_loader, device)
    model.train()
    return evaluate(data_loader.dataset, result_dict, output_dir, iter)


# TODO  test it
def do_train_one_epoch(model,data_loader, optimizer, scheduler, device, output_dir, epoch):

    model.train()
    summary_writer = SummaryWriter(os.path.join(output_dir,"train_logs"))
    for iteration , (images, targets, index) in enumerate(data_loader, (epoch-1)*len(data_loader)):
        batch_time_start = time.time()
        images = images.cuda()
        loss_dict = model(images, targets)
        loss = sum(loss  for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_time_end = time.time()
        time_total = batch_time_end - batch_time_start
        global_step = iteration
        print("\riter:{},cost_time:{:.3f},total_loss:{:.3f},reg_loss:{:.3f},cls_loss:{:.3f} lr:{}".format(iteration,
                                                                                             time_total,
                                                                                                loss,
                                                                                             loss_dict["reg_loss"],
                                                                                             loss_dict["cls_loss"],
                                                                                          scheduler.get_lr()), end=".")
        summary_writer.add_scalar('losses/total_loss', loss, global_step=global_step)
        summary_writer.add_scalar("time-cosumption", time_total, global_step=global_step)
        for loss_name, loss_item in loss_dict.items():
            summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
    print("\n", end="")

    return model


