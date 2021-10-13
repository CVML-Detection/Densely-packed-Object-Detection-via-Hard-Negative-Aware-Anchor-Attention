import torch
from torchvision.ops.boxes import nms as torchvision_nms
from config import device
import math
import os
import shutil
import csv


def bar_custom(current, total, width=30):
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d byte]" % (current / total * 100, percent_bar, current, total)
    return progress


def cxcy_to_xy(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)


def xy_to_cxcy(xy):

    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=1)


def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)


def find_intersection(set_1, set_2):
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)  # 0 혹은 양수로 만드는 부분
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)  # 둘다 양수인 부분만 존재하게됨!


def detect(pred, coder, opts, max_overlap=0.5, top_k=300, is_demo=False):
    pred_bboxes, pred_scores = coder.post_processing(pred, is_demo)
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(0, opts.num_classes):
        class_scores = pred_scores[:, c]
        idx = class_scores > opts.conf_thres

        if idx.sum() == 0:
            continue

        class_scores = class_scores[idx]
        class_bboxes = pred_bboxes[idx]

        sorted_scores, idx_scores = class_scores.sort(descending=True)
        sorted_boxes = class_bboxes[idx_scores]

        # NMS
        num_boxes = len(sorted_boxes)
        keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=max_overlap)
        keep_ = torch.zeros(num_boxes, dtype=torch.bool)
        keep_[keep_idx] = 1
        keep = keep_

        image_boxes.append(sorted_boxes[keep])
        image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
        image_scores.append(sorted_scores[keep])

    if len(image_boxes) == 0:
        image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
        image_labels.append(torch.LongTensor([opts.num_classes]).to(device))  # background
        image_scores.append(torch.FloatTensor([0.]).to(device))

    image_boxes = torch.cat(image_boxes, dim=0)
    image_labels = torch.cat(image_labels, dim=0)
    image_scores = torch.cat(image_scores, dim=0)
    n_objects = image_scores.size(0)

    if n_objects > top_k:
        image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
        image_scores = image_scores[:top_k]
        image_boxes = image_boxes[sort_ind][:top_k]
        image_labels = image_labels[sort_ind][:top_k]

    return image_boxes, image_labels, image_scores


def copy_and_add_header_anno(data_root, eval_gt_path, eval_no_hd_gt_path, split):
    if not os.path.isfile(eval_gt_path):
        gt_path = os.path.join(data_root, 'SKU110K_fixed', 'annotations', 'annotations_{}.csv'.format(split))
        shutil.copy(gt_path, eval_no_hd_gt_path)
        # add header
        with open(eval_no_hd_gt_path, 'r', newline='') as csv_in_file:
            with open(eval_gt_path, 'w', newline='') as csv_out_file:
                freader = csv.reader(csv_in_file)
                fwriter = csv.writer(csv_out_file)
                header_list = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height']
                fwriter.writerow(header_list)
                print(header_list)
                for row in freader:
                    fwriter.writerow(row)

        os.remove(eval_no_hd_gt_path)
        print('copy annotations_{}.csv and add header for evals'.format(split))
