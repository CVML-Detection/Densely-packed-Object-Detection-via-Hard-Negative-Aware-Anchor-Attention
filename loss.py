import torch
import torch.nn as nn
from utils import cxcy_to_xy
from config import device
import math
from sklearn.utils.extmath import cartesian
import numpy as np
import torch.nn.functional as F


# WHD refers https://github.com/javiribera/locating-objects-without-bboxes
def generalize_mean(tensor, dim, p=-1, keepdim=False):
    assert p < 0
    res = torch.mean((tensor + 1e-6) ** p, dim, keepdim=keepdim) ** (1. / p)
    return res


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def cdist(x, y):
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances


class AdvancedWeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height=100,
                 resized_width=100,
                 p=-1):

        super().__init__()

        self.bce = nn.BCELoss()
        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())
        self.p = p

    def set_init(self, resized_height, resized_width):
        self.height = resized_height
        self.width = resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())

    def map2coord(self, map, thres=1.0):
        # gt_map : [B, anchors]
        batch_size = map.size(0)
        mask_100_ = map.reshape(batch_size, -1)  # [B, 10000]
        mask_100 = (mask_100_ >= thres).type(torch.float32)  # [0, 1] 로 바꿔버리기

        nozero_100 = []
        batch_matrices_100 = []

        for b in range(batch_size):
            nozero_100.append(mask_100[b].nonzero().squeeze())
            coordinate_matrix_100 = torch.from_numpy(cartesian([np.arange(self.height), np.arange(self.width)]))
            batch_matrices_100.append(coordinate_matrix_100)

        coordinate_matries_100 = torch.stack(batch_matrices_100, dim=0)
        mask_100_vis = mask_100.view(-1, self.height, self.width)

        # make seq gt
        seq_100 = []
        for b in range(batch_size):
            seq_100.append(coordinate_matries_100[b][nozero_100[b]].to(device))
        return seq_100, mask_100_vis

    def forward(self, prob_map, gt_map):

        gt, mask_100_vis = self.map2coord(map=gt_map)
        orig_sizes = torch.LongTensor([[self.height, self.width], [self.height, self.width]]).to(device)
        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s' \
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated) * self.max_dist + p_replicated * d_matrix

            # our method
            term_2 = torch.mean(torch.min(weighted_d_matrix, 0)[0])

            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        res = terms_1.mean() + terms_2.mean()
        return res


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred_cls, gt_cls):
        alpha_factor = torch.ones_like(gt_cls).to(device) * self.alpha
        a_t = torch.where((gt_cls == 1), alpha_factor, 1. - alpha_factor)
        p_t = torch.where(gt_cls == 1, pred_cls, 1 - pred_cls)
        bce = self.bce(pred_cls, gt_cls)
        cls_loss = a_t * (1 - p_t) ** self.gamma * bce
        return cls_loss


class RetinaLoss(nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.coder = coder
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()

    def giou_loss(self, boxes1, boxes2):

        # iou loss
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [2, s, s, 3]
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [2, s, s, 3]

        inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])                          # [B, s, s, 3, 2]
        inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])                       # [B, s, s, 3, 2]

        inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))  # [B, s, s, 3, 2]
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area                                  # [B, s, s, 3]
        ious = 1.0 * inter_area / union_area                                                 # [B, s, s, 3]

        outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])                          # [B, s, s, 3, 2]
        outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])                       # [B, s, s, 3, 2]
        outer_section = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
        outer_area = outer_section[..., 0] * outer_section[..., 1]                           # [B, s, s, 3]

        giou = ious - (outer_area - union_area)/outer_area
        giou_loss = 1 - giou

        return giou_loss

    def map2mask(self, obj_map, size_100, size_50, anchor_size):
        # expand obj_map to retina mask
        batch_size = obj_map.size(0)
        obj_map = obj_map.detach()
        obj_map_50 = F.interpolate(obj_map.unsqueeze(1), size=size_50)
        obj_map_90000 = obj_map.unsqueeze(-1).expand([obj_map.size(0), size_100[0], size_100[1], 9])
        obj_map_22500 = obj_map_50.squeeze(1).unsqueeze(-1).expand([obj_map.size(0), size_50[0], size_50[1], 9])

        masks = []
        for b in range(batch_size):
            obj_map_90000_ = obj_map_90000[b].reshape(9 * size_100[0] * size_100[1])
            obj_map_22500_ = obj_map_22500[b].reshape(9 * size_50[0] * size_50[1])
            remnant = anchor_size - (9 * size_100[0] * size_100[1] + 9 * size_50[0] * size_50[1])
            zeros_7587 = torch.zeros([remnant]).to(device)
            mask = torch.cat([obj_map_90000_, obj_map_22500_, zeros_7587], dim=0)
            masks.append(mask)

        masks = torch.stack(masks, dim=0)
        return masks

    def hard_negative_aware_anchor_attention(self, loc_mask, niou_mask, n=2):
        # HNAA attention mask
        loc_mask = torch.exp(loc_mask ** n) * loc_mask
        HNAA_mask = loc_mask * torch.exp(niou_mask).unsqueeze(-1)
        return HNAA_mask

    def forward(self, pred, gt_boxes, gt_labels, obj_map=None, size=(800, 800)):
        pred_cls = pred[0]
        pred_loc = pred[1]

        # sanity check
        self.coder.set_anchors(size)
        n_priors = self.coder.center_anchor.size(0)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        # build targets
        gt_cls, gt_locs, depth, depth_ = self.coder.build_target(gt_boxes, gt_labels, IT=0.5)

        h, w = size
        pyramid_levels = np.array([3, 4, 5, 6, 7])
        feature_maps_y = [(h + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        feature_maps_x = [(w + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        size_100 = (feature_maps_y[0], feature_maps_x[0])
        size_50 = (feature_maps_y[1], feature_maps_x[1])  # H W

        map_mask = self.map2mask(obj_map, size_100, size_50, anchor_size=pred_cls.size(1))

        # make mask & num_of_pos
        num_of_pos = (depth > 0).sum().float()                    # only foreground
        cls_mask = (depth_ >= 0).unsqueeze(-1).expand_as(gt_cls)  # both fore and back ground
        loc_mask = torch.where(depth > 0, depth, (depth > 0).type(torch.float32)).unsqueeze(-1)

        # HNAA attention mask
        loc_mask = self.hard_negative_aware_anchor_attention(loc_mask, map_mask)

        # cls loss
        cls_loss = self.focal_loss(pred_cls, gt_cls)

        # loc loss
        gt_boxes = cxcy_to_xy(self.coder.decode(gt_locs.squeeze(0)))
        pred_boxes = cxcy_to_xy(self.coder.decode(pred_loc.squeeze(0))).clamp(0, 1)
        loc_loss = self.giou_loss(pred_boxes, gt_boxes)

        # masking
        cls_loss = (cls_loss * cls_mask).sum() / num_of_pos
        loc_loss = (loc_loss * loc_mask.squeeze()).sum() / num_of_pos
        return cls_loss, loc_loss


class IntegratedLoss(nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.retina_loss = RetinaLoss(coder)
        self.awhd_100 = AdvancedWeightedHausdorffDistance(resized_width=100,
                                                          resized_height=100)

    def forward(self, pred, b_boxes, b_labels, gt_center, gt_cnt, gt_map, size):
        pred_d = pred[:2]
        obj_map = pred[2]

        cls_loss, loc_loss = self.retina_loss(pred_d, b_boxes, b_labels, obj_map, size)
        self.awhd_100.set_init(obj_map.size(1), obj_map.size(2))

        gt_map = F.interpolate(gt_map.unsqueeze(1), size=obj_map.size()[1:]).squeeze(1)
        gt_map = (gt_map - gt_map.min()) / (gt_map.max() - gt_map.min())
        wgd_loss = self.awhd_100(obj_map, gt_map)
        obj_loss = wgd_loss * 0.01
        total_loss = cls_loss + loc_loss + obj_loss
        return total_loss, (cls_loss, loc_loss, obj_loss, obj_loss)


