import os
import torch
import torch.utils.data
import xml.etree.ElementTree as ET
import cv2
import numpy
import utils.setting_dict



class VocDataSet(torch.utils.data.Dataset):

    def __init__(self, data_dir , transform=None):
        self.class_name = utils.setting_dict.class_name
        self.data_dir = data_dir
        self.transform = transform
        self.imagesList = os.listdir(data_dir +"/" + "images/")
        self.annoList = [ image.replace(".jpg", ".xml") for image in self.imagesList]
        self.class_dict = { item.lower() : i for i, item in enumerate(self.class_name)}
        self.imagesList = [os.path.join(data_dir,"images",image)for image in self.imagesList]
        self.annoList = [os.path.join(data_dir,"annotation",anno) for anno in self.annoList]

    def _get_annotation(self, file_path):
        objects = ET.parse(file_path).findall("object")
        boxes = []
        labels = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            boxes.append([x1,y1, x2,y2])
            labels.append(self.class_dict[class_name])
        return (numpy.array(boxes, dtype=numpy.float32), numpy.array(labels, dtype=numpy.float32))

    def get_annotation(self, index):
        return self._get_annotation(self.annoList[index])
    def get_imageInfo(self, index):
        image = self._get_image(self.imagesList[index])
        return image.shape[:2]

    def _get_image(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        image_file = self.imagesList[index]
        anno_file = self.annoList[index]
        image = self._get_image(image_file)
        boxes, labels = self._get_annotation(anno_file)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, {"boxes" :boxes, "labels" : labels}, index

    def __len__(self):
        return len(self.imagesList)

