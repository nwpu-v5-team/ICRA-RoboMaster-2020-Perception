

from data.transform import build_transforms
from .evaluation.voc import voc_evaluation
from .voc import VocDataSet
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate
import torch
import numpy

class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets =  {
                "boxes" : [target["boxes"] for target in list_targets],
                "labels": [target["labels"] for target in list_targets]
            }
        else:
            targets = None
        return images, targets, img_ids



def evaluate(dataset, prediction, output_dir, iter):
    return voc_evaluation(dataset, prediction, output_dir, iter)



def build_dataset(dataset_list, transform=None, is_train = True):
    datasets = []
    for dataset_name in dataset_list:
        dataset = VocDataSet(dataset_name, transform)
        datasets.append(dataset)
    return ConcatDataset(datasets) if is_train else datasets[0]




def make_dataLoader(setting_dict, is_train):
    train_transform =  build_transforms(setting_dict["transform"], is_train)
    dataset_list = setting_dict["data_set"]
    batch_size =   setting_dict["batch_size"]
    dataset = build_dataset(dataset_list,train_transform, is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=BatchCollator(is_train))


