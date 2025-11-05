import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union
from omegaconf import DictConfig
from metrics.agg_mean import AggMean
from omegaconf import OmegaConf
from torch_iou3d import compute_iou3d, compute_distance_coef_diou_loss


def torch_calculate_cube_corners(center, size):
    s = size / 2
    # (B, Classes, 6)
    corners = [
        center + s * torch.FloatTensor([1.0, 1.0, 1.0]),
        center + s * torch.FloatTensor([1.0, 1.0, -1.0]),
        center + s * torch.FloatTensor([1.0, -1.0, 1.0]),
        center + s * torch.FloatTensor([1.0, -1.0, -1.0]),
        center + s * torch.FloatTensor([-1.0, 1.0, 1.0]),
        center + s * torch.FloatTensor([-1.0, 1.0, -1.0]),
        center + s * torch.FloatTensor([-1.0, -1.0, 1.0]),
        center + s * torch.FloatTensor([-1.0, -1.0, -1.0]),
    ]

    return corners

def points_inside_boxes(points, boxes):
    """
    points (B, 117, 3)
    boxes (B, 117, 6): (center x, y, z, size w, h, l)
    """
    return torch.all(points >= boxes[:, :, :3] - boxes[:, :, 3:6] / 2, dim=2) & torch.all(points <= boxes[:, :, :3] + boxes[:, :, 3:6] / 2, dim=2)


def smooth_l1_loss(diff, beta):
    if beta < 1e-5:
        loss = torch.abs(diff)
    else:
        n = torch.abs(diff)
        loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    return loss

# TODO: Translate this

class BBoxLoss:
    def __init__(self, cfg: DictConfig, prefix: str, loss_type: str, weight: float):
        super(BBoxLoss, self).__init__()
        self.cfg = cfg
        self.metrics = self.create_metrics()
        self.was_called = False
        self.samples_dices = []
        self.loss_type = loss_type
        self.weight = weight
        self.prefix = prefix

        assert loss_type in ["L1", "diou3d", "iou3d"], "Invalid loss type"

    def reset_metrics(self, mode: str):
        assert mode in ["epoch", "step"], "Invalid mode"
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset(mode)

    def create_metrics(self):
        metrics = {}
        metrics['l1Loss'] = AggMean()
        metrics['totalLoss'] = AggMean()
        metrics['bones_l1Loss'] = AggMean()
        metrics['organs_l1Loss'] = AggMean()
        metrics['iou'] = AggMean()
        metrics['bones_iou'] = AggMean()
        metrics['organs_iou'] = AggMean()
        metrics['in_view_iou'] = AggMean()
        metrics['out_of_view_iou'] = AggMean()
        metrics['avg_count_in_view'] = AggMean()
        metrics['avg_count_out_of_view'] = AggMean()
        metrics['regression_precision'] = AggMean()
        metrics['regression_recall'] = AggMean()
        metrics['regression_F1'] = AggMean()
        metrics['bones_hit_rate_in_view'] = AggMean()
        metrics['organs_hit_rate_in_view'] = AggMean()


        # Create list of bone and organ classes, and validate that they are correct        
        bone_classes = OmegaConf.load("conf/classes.yaml")['bones']
        bone_classes = [k for k, v in sorted(bone_classes.items(), key=lambda item: item[1])]
        organ_classes = [c for c in self.cfg.data.class_names if c not in bone_classes]
        for bone in bone_classes:
            assert bone in self.cfg.data.class_names, f"Bone class {bone} not found in class_names"
        organ_classes.insert(0, "00_other")
        assert bone_classes[0] == "00_other", "Bone classes must start with '00_other'"
        assert len(organ_classes) > 0 and len(bone_classes) > 0, "No bone or organ classes found"
        assert len(organ_classes) + len(bone_classes) - 1 == len(self.cfg.data.class_names), f"Bone and organ classes do not cover all classes: {len(organ_classes) + len(bone_classes) - 1} != {len(self.cfg.data.class_names)}"

        self.bones_class_indices = np.asarray([self.cfg.data.class_names.index(c) for c in bone_classes], dtype=np.int32)
        self.organs_class_indices = np.asarray([self.cfg.data.class_names.index(c) for c in organ_classes], dtype=np.int32)

        return metrics

    def get_iou3d_for_class_subset(self,
                  pred_boxes: torch.FloatTensor,
                  target_boxes: torch.FloatTensor,
                  box_gt_mask: torch.BoolTensor,
                  class_indices: torch.LongTensor):
        iou = compute_iou3d(pred_boxes[:, class_indices-1, :], target_boxes[:, class_indices-1, :])
        box_gt_mask_sub = box_gt_mask[:, class_indices-1]
        iou = box_gt_mask_sub * iou
        iou = torch.sum(iou) / (torch.sum(box_gt_mask_sub) + 1e-6)
        return iou
    
    def get_iou3d_for_mask(self,
                          pred_boxes: torch.FloatTensor,
                          target_boxes: torch.FloatTensor,
                          mask: torch.BoolTensor):
        iou = compute_iou3d(pred_boxes, target_boxes)
        iou = iou * mask
        iou = torch.sum(iou) / (torch.sum(mask) + 1e-6)
        return iou

    def __call__(self,
                 pred_boxes: torch.FloatTensor,
                 target_boxes: torch.FloatTensor,
                 target_classes: torch.LongTensor):
        # pred_boxes: (B, 117, 6)
        # target_boxes: (B, 117, 6)
        # target_classes: (B, 117)
        # We always predict 117 boxes, since each class corresponds to a single box.
        # However, sometimes a box is not present in the GT, so target_classes it 0 in that case.
        # We need to ignore these boxes in the loss.
        # NOTE: we always try to predict out-of-window boxes. But, it's a separate metric.
        l1_loss = torch.mean(smooth_l1_loss(pred_boxes - target_boxes, 1.0), dim=2)
        box_gt_mask = target_classes != 0
        l1_loss = l1_loss * box_gt_mask

        B = pred_boxes.shape[0]
        bones_l1Loss = torch.mean(l1_loss[:, self.bones_class_indices - 1])
        organs_l1Loss = torch.mean(l1_loss[:, self.organs_class_indices - 1])

        totalLoss = l1_loss
        totalLoss = torch.mean(totalLoss)

        iou = compute_iou3d(pred_boxes, target_boxes)
        iou = box_gt_mask * iou.reshape(B, -1)
        iou = torch.sum(iou, dim=-1) / (torch.sum(box_gt_mask, dim=-1) + 1e-5)
        iou = torch.mean(iou)

        if self.loss_type == "iou3d":
            totalLoss = totalLoss + 1.0 - iou
        elif self.loss_type == "diou3d":
            totalLoss = totalLoss + 1.0 - iou
            dist = compute_distance_coef_diou_loss(pred_boxes, target_boxes)
            dist = box_gt_mask * dist.reshape(B, -1)
            dist = torch.sum(dist, dim=-1) / (torch.sum(box_gt_mask, dim=-1) + 1e-5)
            dist = torch.mean(dist)
            totalLoss = totalLoss + dist

        totalLoss *= self.weight  # Fixed weight to balance with the segmentation loss

        with torch.no_grad():
            self.metrics['bones_l1Loss'].update(bones_l1Loss.item(), B)
            self.metrics['organs_l1Loss'].update(organs_l1Loss.item(), B)
            self.metrics['l1Loss'].update(torch.mean(l1_loss).item(), B)
            self.metrics['totalLoss'].update(totalLoss.item(), B)
            self.metrics['iou'].update(iou.item(), B)
            self.metrics['bones_iou'].update(
                self.get_iou3d_for_class_subset(pred_boxes, target_boxes, box_gt_mask, self.bones_class_indices).item(), B
            )
            self.metrics['organs_iou'].update(
                self.get_iou3d_for_class_subset(pred_boxes, target_boxes, box_gt_mask, self.organs_class_indices).item(), B
            )

            # in-view bounding boxes = those whose distance to the center is less than or equal to 1.0
            center_in_view_mask = torch.sum((target_boxes[:, :, :3] ** 2), dim=2) <= 1.0
            out_of_view_mask = (~center_in_view_mask) | (~box_gt_mask)
            in_view_mask = ~out_of_view_mask

            self.metrics['in_view_iou'].update(
                self.get_iou3d_for_mask(pred_boxes, target_boxes, in_view_mask * box_gt_mask).item(), B
            )

            self.metrics['out_of_view_iou'].update(
                self.get_iou3d_for_mask(pred_boxes, target_boxes, out_of_view_mask * box_gt_mask).item(), B
            )

            self.metrics['avg_count_in_view'].update(
                torch.mean(torch.sum(in_view_mask, dim=1).float()).item(), B
            )

            self.metrics['avg_count_out_of_view'].update(
                torch.mean(torch.sum(out_of_view_mask, dim=1).float()).item(), B
            )

            in_view_pred = torch.sum((pred_boxes[:, :, :3] ** 2), dim=2) <= 1.0

            FP = torch.sum(in_view_pred * out_of_view_mask)
            FN = torch.sum((~in_view_pred) * in_view_mask)
            TP = torch.sum(in_view_pred * in_view_mask)

            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)

            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            self.metrics['regression_precision'].update(precision.item(), B)
            self.metrics['regression_recall'].update(recall.item(), B)
            self.metrics['regression_F1'].update(f1.item(), B)

            pred_centers = pred_boxes[:, :, :3]
            # Check when pred_centers are inside the GT boxes
            pred_inside_gt = points_inside_boxes(pred_centers, target_boxes)
            pred_inside_gt = pred_inside_gt * in_view_mask * box_gt_mask
            
            # -1 because we don't have a prediction for the background class, so the class indices have a 1-offset
            bones_hit_rate_in_view = torch.sum(pred_inside_gt[:, self.bones_class_indices-1]) / (torch.sum(in_view_mask[:, self.bones_class_indices-1]) + 1e-6)
            organs_hit_rate_in_view = torch.sum(pred_inside_gt[:, self.organs_class_indices-1]) / (torch.sum(in_view_mask[:, self.organs_class_indices-1]) + 1e-6)

            self.metrics['bones_hit_rate_in_view'].update(bones_hit_rate_in_view.item(), B)
            self.metrics['organs_hit_rate_in_view'].update(organs_hit_rate_in_view.item(), B)


            
        return totalLoss