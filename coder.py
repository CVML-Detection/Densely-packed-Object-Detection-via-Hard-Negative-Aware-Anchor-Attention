import torch
from abc import ABCMeta, abstractmethod
from config import device
from utils import find_jaccard_overlap, xy_to_cxcy
from anchor import RETINA_Anchor
from utils import cxcy_to_xy


class Coder(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class RETINA_Coder(Coder):
    def __init__(self, opts):
        super().__init__()
        self.data_type = opts.data_type
        self.anchor_obj = RETINA_Anchor('retina')
        self.num_classes = opts.num_classes
        self.anchor_dic = {}

    def set_anchors(self, size):
        if self.anchor_dic.get(size) is None:
            self.anchor_dic[size] = self.anchor_obj.create_anchors(img_size=size)
        self.center_anchor = self.anchor_dic[size]

    def assign_anchors_to_device(self):
        self.center_anchor = self.center_anchor.to(device)

    def assign_anchors_to_cpu(self):
        self.center_anchor = self.center_anchor.to('cpu')

    def encode(self, cxcy):
        gcxcy = (cxcy[:, :2] - self.center_anchor[:, :2]) / self.center_anchor[:, 2:]
        gwh = torch.log(cxcy[:, 2:] / self.center_anchor[:, 2:])

        return torch.cat([gcxcy, gwh], dim=1)

    def decode(self, gcxgcy):
        cxcy = gcxgcy[:, :2] * self.center_anchor[:, 2:] + self.center_anchor[:, :2]
        wh = torch.exp(gcxgcy[:, 2:]) * self.center_anchor[:, 2:]
        return torch.cat([cxcy, wh], dim=1)

    # IT - IoU Threshold == 0.5
    def build_target(self, gt_boxes, gt_labels, IT=None):
        batch_size = len(gt_labels)
        n_priors = self.center_anchor.size(0)

        # ----- 1. make container
        gt_locations = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, device=device)
        gt_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float, device=device)
        anchor_identifier = -1 * torch.ones((batch_size, n_priors), dtype=torch.float32, device=device)
        anchor_identifier_1 = -1 * torch.ones((batch_size, n_priors), dtype=torch.float32, device=device)
        # if anchor is positive -> 1,
        #              negative -> 0,
        #              ignore   -> -1

        # ----- 2. make corner anchors
        corner_anchor = cxcy_to_xy(self.center_anchor)

        for i in range(batch_size):
            boxes = gt_boxes[i]
            labels = gt_labels[i]

            # ----- 3. *** normalized_iou_assign ***
            iou = find_jaccard_overlap(corner_anchor, boxes)
            IoU_max, IoU_argmax = iou.max(dim=1)
            IoU_max_per_obj, _ = iou.max(dim=0)
            Normed_IoU_max = IoU_max / IoU_max_per_obj[IoU_argmax]

            # ----- 4-1. build gt_classes
            negative_indices = Normed_IoU_max < 0.7
            negative_indices_1 = IoU_max < 0.4
            gt_classes[i][negative_indices, :] = 0
            anchor_identifier[i][negative_indices] = 0
            anchor_identifier_1[i][negative_indices_1] = 0

            if IT is not None:
                positive_indices = Normed_IoU_max >= 0.7
                positive_indices_1 = IoU_max >= 0.5
            else:
                _, IoU_argmax_per_object = iou.max(dim=0)
                positive_indices = torch.zeros_like(IoU_max)
                positive_indices[IoU_argmax_per_object] = 1
                positive_indices = positive_indices.type(torch.bool)

            argmax_labels = labels[IoU_argmax]
            gt_classes[i][positive_indices, :] = 0
            gt_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.
            anchor_identifier[i][positive_indices] = Normed_IoU_max[positive_indices]
            anchor_identifier_1[i][positive_indices_1] = 1

            # ----- 4-2. build gt_locations
            argmax_locations = boxes[IoU_argmax]
            center_locations = xy_to_cxcy(argmax_locations)
            gt_gcxcywh = self.encode(center_locations)
            gt_locations[i] = gt_gcxcywh
        return gt_classes, gt_locations, anchor_identifier, anchor_identifier_1

    def post_processing(self, pred, is_demo=False):
        if is_demo:
            self.assign_anchors_to_cpu()
            pred_cls = pred[0].to('cpu')
            pred_loc = pred[1].to('cpu')
        else:
            pred_cls = pred[0]
            pred_loc = pred[1]

        n_priors = self.center_anchor.size(0)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)

        pred_bboxes = cxcy_to_xy(self.decode(pred_loc.squeeze())).clamp(0, 1)
        pred_scores = pred_cls.squeeze(0)
        return pred_bboxes, pred_scores


if __name__ == '__main__':
    ssd_coder = RETINA_Coder()
    ssd_coder.assign_anchors_to_device()
    print(ssd_coder.center_anchor)