import logging
import os
from datetime import datetime

import numpy as np

from .eval_voc import eval_detection_voc
import utils.setting_dict


def voc_evaluation(dataset, predictions, output_dir, iteration=None):
    class_names = dataset.class_name

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []

    #print(predictions)
    for i in range(len(dataset)):
        annotation = dataset.get_annotation(i)
        #print(annotation)
        gt_boxes, gt_labels = annotation
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)

        img_size = dataset.get_imageInfo(i)
        prediction = predictions[i]
        boxes, labels, scores = prediction['boxes'], prediction['indexs'], prediction['scores']


        boxes[:, 0::2] = boxes[:, 0::2] * img_size[1]
        boxes[:, 1::2] = boxes[:, 1::2] * img_size[0]
        # print(boxes)
        # print(gt_boxes)

        pred_boxes_list.append(boxes.cpu().numpy())
        pred_labels_list.append(labels.cpu().numpy())
        pred_scores_list.append(scores.cpu().numpy())
    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=None,
                                iou_thresh=0.5,
                                use_07_metric=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SSD.inference")
    logger.setLevel(logging.INFO)

    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        metrics[class_names[i]] = ap
        result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    #logger.info(result_str)
    print(result_str)

    if iteration is not None:
        result_path = output_dir + '/result_{:07d}.txt'.format(iteration)
        #print("*************")
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return dict(metrics=metrics)
