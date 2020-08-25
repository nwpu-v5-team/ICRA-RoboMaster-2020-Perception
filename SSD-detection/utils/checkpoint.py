import logging
import os
import torch



class CheckPoint:
    last_check_point = "last_checkpoint.txt"
    def __init__(self, model, optimizer=None, scheduler=None, save_dir="", logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir =save_dir
        self.logger = logger


    def has_checkpoint(self):
        return os.path.exists(os.path.join(self.save_dir, self.last_check_point))

    def tag_checkpoint(self, name):
        with open(os.path.join(self.save_dir, self.last_check_point), "a+") as tag_file:
            tag_file.write("{}\n".format(name))
    def get_last_name(self):
        last_name = ""
        with open(os.path.join(self.save_dir, self.last_check_point), "r") as tag_file:
             last_name = tag_file.readlines()[-1]
        return last_name

    def _load(self, name):
        file_path = os.path.join(self.save_dir, "{}.pth".format(name))
        return torch.load(file_path, map_location=torch.device("cpu"))
    def save(self, name):
        data = {}
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["scheduler"] = self.scheduler

        self.logger.info("Save checkPoint{}".format(self.save_dir +'/'+ name))
        torch.save(data, os.path.join(self.save_dir, "{}.pth".format(name)))
        self.tag_checkpoint(name)


    def load(self, use_last = True):
        self.logger.info("Load checkPoint")
        if use_last and self.has_checkpoint():
            last_name = self.get_last_name()
            checkpoint = self._load(last_name)
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            self.model.load_state_dict(checkpoint.pop("model"))

            return checkpoint
        else :
            return None
