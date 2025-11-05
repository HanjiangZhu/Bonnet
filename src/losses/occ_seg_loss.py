# Same as seg_loss, but points have shape (B, N, 3), labels have shape (B, N) and thus naturally supports masks.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union
import functools
from omegaconf import DictConfig
from metrics.agg_mean import AggMean
from metrics.agg_cfmat import AggCfMatrix
from omegaconf import OmegaConf
from losses.base_loss import BaseLoss

SMOOTH = 1e-5
"""
Takes in labels of shape (num_points) and returns a one-hot encoding of shape (num_points, num_classes).
"""
def numpy_onehot(labels, num_classes):
    onehot = np.eye(num_classes)[labels]
    return onehot

def masked_global_multiclass_dice(a_onehot, b_onehot, mask, include_bg=False):
    """
    Expects a_onehot and b_onehot to be of shape (batch_size * num_points, num_classes)
    Returns the overall multiclass dice
    """
    # Multiply mask by the onehot encodings
    a_onehot = a_onehot * mask[:, None] # Shape (batch_size * num_points, num_classes)
    b_onehot = b_onehot * mask[:, None] # Shape (batch_size * num_points, num_classes)

    if not include_bg:
        a_onehot = a_onehot[:, 1:]
        b_onehot = b_onehot[:, 1:]

    intersect = torch.sum(a_onehot * b_onehot, dim=0) # Shape (num_classes)
    fsum = torch.sum(a_onehot, dim=0) # Shape (num_classes)
    ssum = torch.sum(b_onehot, dim=0) # Shape (num_classes)
    dice = (2 * intersect + SMOOTH) / (fsum + ssum + SMOOTH) # Shape (num_classes)
    return torch.mean(dice)

"""
Computes the average dice accross all classes and all samples in the batch
Expects a_onehot and b_onehot to be of shape (batch_size, num_points, num_classes)
Mask is of shape (batch_size, num_points) and is a boolean mask
Returns the per-sample multiclass dice
"""
def masked_per_sample_multiclass_dice(pred_onehot, target_onehot, mask,
                                      empty_gt_mode="remove_from_mean", return_mode="mean"):
    assert empty_gt_mode in ["set_to_one", "remove_from_mean"], "empty_gt_mode must be either 'set_to_one' or 'remove_from_mean'"

    # Multiply mask by the onehot encodings
    pred_onehot = pred_onehot * mask[:, :, None] # Shape (batch_size, num_points, num_classes)
    target_onehot = target_onehot * mask[:, :, None] # Shape (batch_size, num_points, num_classes)

    intersect = torch.sum(pred_onehot * target_onehot, dim=1) # Shape (batch_size, num_classes)
    pred_sum = torch.sum(pred_onehot, dim=1) # Shape (batch_size, num_classes)
    target_sum = torch.sum(target_onehot, dim=1) # Shape (batch_size, num_classes)

    if empty_gt_mode == "remove_from_mean":
        dice = (2 * intersect) / (pred_sum + target_sum + SMOOTH) # Shape (batch_size, num_classes)
        # The negative dices are the ones where the ground truth is empty
        valid_dices_for_each_sample = torch.sum(pred_sum + target_sum > 0, dim=1) # Shape (batch_size)
        sum_dices_for_each_sample = torch.sum(dice, dim=1) # Shape (batch_size)
        # Final dice is the average of the valid dices
        dice = sum_dices_for_each_sample / valid_dices_for_each_sample
    else: # set_to_one
        dice = (2 * intersect + SMOOTH) / (pred_sum + target_sum + SMOOTH) # Shape (batch_size, num_classes)
        dice = torch.mean(dice, dim=(1))
    
    if return_mode == "mean":
        dice = torch.mean(dice)
    else:
        assert return_mode == "list", "return_mode must be either 'mean' or 'list'"
        # in this case we return a list of dices for each sample in the batch

    return dice

"""
Expects a_onehot and b_onehot to be of shape (num_points, num_classes)
Returns the global multiclass dice
pnts_to_use is a boolean mask of shape (num_points) that indicates which points to use in the calculation
"""
def masked_numpy_flat_multiclass_dice(a_onehot, b_onehot, pnts_to_use):
    assert pnts_to_use.shape[0] == a_onehot.shape[0], "pnts_to_use must have the same number of points as a_onehot"

    # Multiply mask by the onehot encodings
    a_onehot = a_onehot * pnts_to_use[:, None] # Shape (num_points, num_classes)
    b_onehot = b_onehot * pnts_to_use[:, None] # Shape (num_points, num_classes)

    intersect = np.sum(a_onehot * b_onehot, axis=0) # Shape (num_classes)
    fsum = np.sum(a_onehot, axis=0) # Shape (num_classes)
    ssum = np.sum(b_onehot, axis=0) # Shape (num_classes)
    dice = (2 * intersect + SMOOTH) / (fsum + ssum + SMOOTH) # Shape (num_classes)
    return np.mean(dice)

def get_label_subset(labels, class_subset_indices):
    # labels is of shape (batch_size, num_points)
    # We set every prediction that is not in class_subset_indices to 0
    # This is useful for calculating the dice of a subset of classes
    mask = torch.isin(labels, class_subset_indices, invert=True)
    labels = torch.clone(labels)
    labels[mask] = 0
    return labels

class OccupancySegmentationLoss(BaseLoss):
    def __init__(self,
                 cfg: DictConfig, prefix: str,
                 label_smoothing: float = 0.0):
        super(OccupancySegmentationLoss, self).__init__()
        self.cfg = cfg
        self.metrics = self.create_metrics()
        self.was_called = False
        self.samples_dices = []
        self.prefix = prefix
        self.label_smoothing = label_smoothing

    def create_metrics(self):
        metrics = {}
        class_names = self.cfg.data.class_names

        metrics['nllLoss'] = AggMean()
        metrics['softDice'] = AggMean()
        metrics['totalLoss'] = AggMean()
        metrics['cfmat'] = AggCfMatrix(class_names, self.cfg.device)

        # Create list of bone and organ classes, and validate that they are correct
        classes = OmegaConf.load("conf/classes.yaml")
        bone_classes = classes['bones']
        bone_classes = [k for k, v in sorted(bone_classes.items(), key=lambda item: item[1])]
        ribs_classes = classes['ribs']
        ribs_classes = [k for k, v in sorted(ribs_classes.items(), key=lambda item: item[1])]
        spine_classes = classes['spine']
        spine_classes = [k for k, v in sorted(spine_classes.items(), key=lambda item: item[1])]
        limbs_classes = classes['limbs']
        limbs_classes = [k for k, v in sorted(limbs_classes.items(), key=lambda item: item[1])]
        
        organ_classes = [c for c in class_names if c not in bone_classes]
        
        spine_classes.insert(0, "00_other")
        ribs_classes.insert(0, "00_other")
        limbs_classes.insert(0, "00_other")
        organ_classes.insert(0, "00_other")
        
        for bone in bone_classes:
            assert bone in class_names, f"Bone class {bone} not found in class_names"
        
        assert bone_classes[0] == "00_other", "Bone classes must start with '00_other'"
        assert len(organ_classes) > 0 and len(bone_classes) > 0, "No bone or organ classes found"
        assert len(organ_classes) + len(bone_classes) - 1 == len(self.cfg.data.class_names), f"Bone and organ classes do not cover all classes: {len(organ_classes) + len(bone_classes) - 1} != {len(self.cfg.data.class_names)}"

        self.bones_class_indices = torch.tensor([class_names.index(c) for c in bone_classes], dtype=torch.int64, device=self.cfg.device)
        self.organs_class_indices = torch.tensor([class_names.index(c) for c in organ_classes], dtype=torch.int64, device=self.cfg.device)
        self.ribs_class_indices = torch.tensor([class_names.index(c) for c in ribs_classes], dtype=torch.int64, device=self.cfg.device)
        self.spine_class_indices = torch.tensor([class_names.index(c) for c in spine_classes], dtype=torch.int64, device=self.cfg.device)
        self.limbs_class_indices = torch.tensor([class_names.index(c) for c in limbs_classes], dtype=torch.int64, device=self.cfg.device)

        metrics['bones_overallDice'] = AggCfMatrix(class_names, device=self.cfg.device, labels=self.bones_class_indices)
        metrics['organs_overallDice'] = AggCfMatrix(class_names, device=self.cfg.device, labels=self.organs_class_indices)
        metrics['ribs_overallDice'] = AggCfMatrix(class_names, device=self.cfg.device, labels=self.ribs_class_indices)
        metrics['spine_overallDice'] = AggCfMatrix(class_names, device=self.cfg.device, labels=self.spine_class_indices)
        metrics['limbs_overallDice'] = AggCfMatrix(class_names, device=self.cfg.device, labels=self.limbs_class_indices)

        return metrics
    
    def on_start_epoch(self):
        super().on_start_epoch()
        self.sample_counter = 0
    
    def on_end_epoch(self):
        super().on_end_epoch()

    @torch.no_grad()
    def sync_gpu_and_update_metrics(self,
                                    nll_loss, soft_dice, total_loss,
                                    pred_y, target, mask, B: int):
        self.metrics['nllLoss'].update(nll_loss, B)
        self.metrics['softDice'].update(soft_dice, B)
        self.metrics['totalLoss'].update(total_loss, B)
        self.metrics['cfmat'].update(pred_y, target)

        mask_bool = (mask > 0).bool()
        bones_target_mask = torch.isin(target, self.bones_class_indices)

        self.metrics['bones_overallDice'].update(get_label_subset(pred_y[bones_target_mask], self.bones_class_indices),
                                                 target[bones_target_mask], mask=mask_bool[bones_target_mask])
        
        organs_target_mask = torch.isin(target, self.organs_class_indices)

        self.metrics['organs_overallDice'].update(get_label_subset(pred_y[organs_target_mask], self.organs_class_indices),
                                                  target[organs_target_mask], mask=mask_bool[organs_target_mask])

        ribs_target_mask = torch.isin(target, self.ribs_class_indices)

        self.metrics['ribs_overallDice'].update(get_label_subset(pred_y[ribs_target_mask], self.ribs_class_indices),
                                                target[ribs_target_mask], mask=mask_bool[ribs_target_mask])
        
        spine_target_mask = torch.isin(target, self.spine_class_indices)

        self.metrics['spine_overallDice'].update(get_label_subset(pred_y[spine_target_mask], self.spine_class_indices),
                                                 target[spine_target_mask], mask=mask_bool[spine_target_mask])
        
        limbs_target_mask = torch.isin(target, self.limbs_class_indices)

        self.metrics['limbs_overallDice'].update(get_label_subset(pred_y[limbs_target_mask], self.limbs_class_indices),
                                                 target[limbs_target_mask], mask=mask_bool[limbs_target_mask])

    # Expects logits to be of shape (batch_size, num_points, num_classes)   
    # Target is of shape (batch_size, num_points)
    # Mask is of shape (batch_size, num_points) and is a boolean mask
    # indicating which points to use in the calculation
    def __call__(self,
                 logits_or_preds: torch.FloatTensor,
                 target: Union[torch.LongTensor, torch.FloatTensor],
                 mask: torch.BoolTensor):
        self.was_called = True

        cfg = self.cfg
        total_loss = 0
        mask_flat = mask.reshape(-1) # Shape (batch_size * num_points)

        B = logits_or_preds.size(0)
        N = logits_or_preds.size(1)
        C = logits_or_preds.size(2)

        assert mask_flat.shape[0] == B * N, f"Mask has incorrect shape: {mask_flat.shape[0]} != {B * N}"
        assert len(target.shape) == 2, f"Target has incorrect shape {target.shape}"
        assert len(logits_or_preds.shape) == 3, f"logits_or_preds has incorrect shape {logits_or_preds.shape}"

        if len(target.shape) == 2: # If target is in format (batch_size, num_points)
            target = target.long()
            target_flat = target.reshape(-1) # Shape (batch_size * num_points)
        else: # If target is in format (batch_size, num_points, num_classes)
            assert len(target.shape) == 3, "Target has incorrect shape"
            target_flat = target.reshape(-1, target.shape[-1])
            target = torch.argmax(target, dim=-1).long()
        
        target_onehot = F.one_hot(target, cfg.data.num_classes).float()
        logits_flat = logits_or_preds.reshape(-1, cfg.data.num_classes) # Shape (batch_size * num_points, num_classes)
        softmax_out = F.softmax(logits_or_preds, dim=-1).reshape(B * N, C)
        
        pred_y = torch.argmax(logits_or_preds, dim=-1).reshape(B, N) # Shape (batch_size, num_points)
        target_flat_onehot = target_onehot.reshape(B * N, C) # Shape (batch_size * num_points, num_classes)

        # Loss function = NLL + SoftDice
        nll_loss = None
        if logits_flat is not None:
            nll_loss = F.cross_entropy(logits_flat, target_flat, label_smoothing=self.label_smoothing, reduction='none')
            nll_loss *= mask_flat
            nll_loss = torch.sum(nll_loss) / (torch.sum(mask_flat) + 1e-7)
            total_loss += nll_loss

        # Soft dice is calculated as the global dice of each batch
        soft_dice = masked_global_multiclass_dice(softmax_out, target_flat_onehot, mask_flat, include_bg=False)
        total_loss += 1 - soft_dice

        self.sync_gpu_and_update_metrics(nll_loss.detach(),
                                         soft_dice.detach(),
                                         total_loss.detach(),
                                         pred_y.detach(),
                                         target.detach(),
                                         mask.detach(), B)
            
        self.sample_counter += B
        return total_loss