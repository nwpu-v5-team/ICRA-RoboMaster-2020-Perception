import torch
from  torchvision.ops import nms
from itertools import product

#
# nms
#
class Processor:
    def __init__(self):
        super().__init__()
        self.NMS_THRESHOLD = 0.2#setting_dict["NMS_THRESHOLD"]
        self.CONF_THRESHOLD = 0.01#setting_dict["CONF_THRESHOLD"]
    def __call__(self, detections):
        results = []
        batch_scores, batch_boxes = detections


        for idx in range(batch_scores.shape[0]):
            scores = batch_scores[idx]
            bboxes = batch_boxes[idx]
            num_boxes = scores.shape[0]
            num_class = scores.shape[-1]

            bboxes = bboxes.view(num_boxes, 1, 4).expand(num_boxes, num_class, 4)


            bboxes = bboxes[:, 1:]
            scores = scores[..., 1:]
            #labels = labels[..., 1:]
            labels = torch.arange(1,num_class, device=batch_scores.device)
            labels = labels.view(1, -1).expand_as(scores)

            boxes = bboxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            indices = torch.nonzero(scores > self.CONF_THRESHOLD).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]
            max_coordinate = 1314
            offsets = labels*(max_coordinate+1)

            nms_boxes = boxes + offsets.reshape(boxes.shape[0],1)

            keep = nms(nms_boxes, scores, self.NMS_THRESHOLD)

            keep = keep[:20]

            results.append( dict(
                    boxes=boxes[keep],
                    scores= scores[keep],
                    indexs = labels[keep])
            )

        return results

class processor2 :
    def __init__(self, prod_threshold, nms_threshold):
        self.prod_threshold = prod_threshold
        self.nms_threshold = nms_threshold
    def __call__(self, detection):
        batch_scores, batch_boxes = detection
        scores = batch_scores[0]
        bboxes = batch_boxes[0]

        scores, indexs = torch.max(scores, 1)


        return (bboxes, scores, indexs)
