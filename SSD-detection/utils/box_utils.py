import torch
import math

#box format => centerx, centery width, height
#    format => lx, ly , rx, ry
#location  centerx, centery width, height( needs var)


def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    return [x1,y1, x2-x1, y2-y1]
def xywh2xyxy(box):
    x1, y1, w,h = box
    return [x1,y1, x1+w, y1+h]
def box2xyxy(box):
    centerx, centery, w, h=box
    return [centerx - w/2, centery - h/2, centerx + w/2, centery + h/2]
def xyxy2box(box):
    x1,y1,x2,y2 = box
    return [(x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1)]

def location2box(locations, priors, center_var, size_var):
    return torch.cat(
        [locations[..., :2]*center_var*priors[..., 2:] + priors[..., :2],
        torch.exp(locations[...,2:]*size_var)*priors[..., 2:] ],
        dim= locations.dim()-1
    )
def box2location(box, prior, center_var, size_var):
    return torch.cat([
        (box[..., :2] - prior[..., :2])/ prior[..., 2:]/center_var,
        torch.log(box[..., 2:]/prior[...,2:])/ size_var],
        dim = box.dim() -1
    )
def center2corner(boxes):
    return torch.cat(
        [
            boxes[...,:2] - boxes[...,2:]/2,
            boxes[...,:2] + boxes[..., 2:]/2
        ],
        dim=boxes.dim()-1
    )
def corner2center(boxes):
    return torch.cat(
        [
            (boxes[...,2:] + boxes[..., :2])/2,
            (boxes[...,2:] - boxes[..., :2])
        ],
        dim=boxes.dim()-1
    )
def area(box):
    return box[2]*box[3]

def iou_center(box1, box2, eps=1e5):
    xyxybox1 = box2xyxy(box1)
    xyxybox2 = box2xyxy(box2)
    xmin, ymin = torch.max(torch.cat([ torch.tensor(xyxybox1[:2]), torch.tensor(xyxybox2[:2])],dim=0) )
    xmax, ymax = torch.min(torch.cat([ torch.tensor(xyxybox1[2:]), torch.tensor(xyxybox2[2:])],dim=0) )
    if xmax < xmin and ymin > ymax :
        return 0
    interArea = area(xyxy2box([xmin, ymin, xmax, ymax]))
    return interArea/(area(box1)+ area(box2)-interArea + eps)
def iou_corner(box1, box2, eps=1e5):
    xyxybox1 = (box1)
    xyxybox2 = (box2)
    xmin, ymin = torch.max(torch.cat([ torch.tensor(xyxybox1[:2]), torch.tensor(xyxybox2[:2])],dim=0) )
    xmax, ymax = torch.min(torch.cat([ torch.tensor(xyxybox1[2:]), torch.tensor(xyxybox2[2:])],dim=0) )
    if xmax < xmin and ymin > ymax :
        return 0
    interArea = area(xyxy2box([xmin, ymin, xmax, ymax]))
    return interArea/(area(box1)+ area(box2)-interArea + eps)

def area_of(left_top, right_bottom) -> torch.Tensor:
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels




# hard negative mining
# this method can suppress the quantity of false negative

def negative_supress(loss, labels, neg_pos_ratio):

    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask