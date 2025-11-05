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

SMOOTH = 1e-10

"""
Takes in labels of shape (num_points) and returns a one-hot encoding of shape (num_points, num_classes).
"""
def numpy_onehot(labels, num_classes):
    onehot = np.eye(num_classes)[labels]
    return onehot

"""
Expects a_onehot and b_onehot to be of shape (num_points, num_classes)
Returns the global multiclass dice
"""
def numpy_multiclass_dice(a_probs, b_probs, include_bg=False):
    assert a_probs.shape == b_probs.shape, f"Shapes do not match: {a_probs.shape} != {b_probs.shape}"
    assert len(a_probs.shape) == 2, f"Input shape is incorrect: {a_probs.shape}"
    # Input shape: (num_points, num_classes)
    a_probs = a_probs[:, 1:] if not include_bg else a_probs
    b_probs = b_probs[:, 1:] if not include_bg else b_probs
    intersect = np.sum(a_probs * b_probs, dim=0) # Shape (num_classes)
    fsum = np.sum(a_probs, dim=0) # Shape (num_classes)
    ssum = np.sum(b_probs, dim=0) # Shape (num_classes)
    dice = (2 * intersect + SMOOTH) / (fsum + ssum + SMOOTH) # Shape (num_classes)
    return np.mean(dice)

def torch_multiclass_dice(a_probs, b_probs, include_bg=False):
    assert a_probs.shape == b_probs.shape, f"Shapes do not match: {a_probs.shape} != {b_probs.shape}"
    assert len(a_probs.shape) == 2, f"Input shape is incorrect: {a_probs.shape}"
    # Input shape: (num_points, num_classes)
    a_probs = a_probs[:, 1:] if not include_bg else a_probs
    b_probs = b_probs[:, 1:] if not include_bg else b_probs
    intersect = torch.sum(a_probs * b_probs, dim=0) # Shape (num_classes)
    fsum = torch.sum(a_probs, dim=0) # Shape (num_classes)
    ssum = torch.sum(b_probs, dim=0) # Shape (num_classes)
    dice = (2 * intersect + SMOOTH) / (fsum + ssum + SMOOTH) # Shape (num_classes)
    return torch.mean(dice)

def get_label_subset(labels, class_subset_indices):
    # labels is of shape (batch_size, num_points)
    # We set every prediction that is not in class_subset_indices to 0
    # This is useful for calculating the dice of a subset of classes
    mask = torch.isin(labels, class_subset_indices, invert=True)
    labels = torch.clone(labels)
    labels[mask] = 0
    return labels

class SegmentationLoss(BaseLoss):
    def __init__(self, cfg: DictConfig,
                 label_smoothing: float,
                 disable_metrics: bool=False):
        super(SegmentationLoss, self).__init__()
        self.disable_metrics = disable_metrics
        self.cfg = cfg
        self.metrics = self.create_metrics()
        self.sample_counter = 0
        self.label_smoothing = label_smoothing

    def create_metrics(self):
        metrics = {}
        
        if self.disable_metrics:
            return metrics
        
        class_names = self.cfg.data.class_names

        metrics['nllLoss'] = AggMean()
        metrics['softDice'] = AggMean()
        metrics['totalLoss'] = AggMean()
        metrics['cfmat'] = AggCfMatrix(class_names, self.cfg.device)

        # Create list of bone and organ classes, and validate that they are correct
        classes = OmegaConf.load("../conf/classes.yaml")
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
                                    pred_y, target, B: int):
        self.metrics['nllLoss'].update(nll_loss, B)
        self.metrics['softDice'].update(soft_dice, B)
        self.metrics['totalLoss'].update(total_loss, B)

        self.metrics['cfmat'].update(pred_y, target)

        bones_target_mask = torch.isin(target, self.bones_class_indices)

        self.metrics['bones_overallDice'].update(get_label_subset(pred_y[bones_target_mask], self.bones_class_indices),
                                                 target[bones_target_mask])
        
        organs_target_mask = torch.isin(target, self.organs_class_indices)

        self.metrics['organs_overallDice'].update(get_label_subset(pred_y[organs_target_mask], self.organs_class_indices),
                                                  target[organs_target_mask])

        ribs_target_mask = torch.isin(target, self.ribs_class_indices)

        self.metrics['ribs_overallDice'].update(get_label_subset(pred_y[ribs_target_mask], self.ribs_class_indices),
                                                target[ribs_target_mask])
        
        spine_target_mask = torch.isin(target, self.spine_class_indices)

        self.metrics['spine_overallDice'].update(get_label_subset(pred_y[spine_target_mask], self.spine_class_indices),
                                                 target[spine_target_mask])
        
        limbs_target_mask = torch.isin(target, self.limbs_class_indices)

        self.metrics['limbs_overallDice'].update(get_label_subset(pred_y[limbs_target_mask], self.limbs_class_indices),
                                                 target[limbs_target_mask])

    # Expects logits to be of shape (batch_size, num_points, num_classes)   
    # Target is of shape (batch_size, num_points)
    # indicating which points to use in the calculation
    def __call__(self,
                 logits: torch.FloatTensor,
                 target: Union[torch.LongTensor, torch.FloatTensor],
                 batch_size: int):
        total_loss = 0
        B = batch_size

        assert len(target.shape) == 1, f"Target has incorrect shape {target.shape}"
        assert len(logits.shape) == 2, f"Logits has incorrect shape {logits.shape}"
        
        target_onehot = F.one_hot(target, num_classes=self.cfg.data.num_classes).float()
        softmax_out = F.softmax(logits, dim=-1)

        nll_loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        soft_dice = torch_multiclass_dice(softmax_out, target_onehot, include_bg=False)

        total_loss += nll_loss
        total_loss += 1 - soft_dice

        pred_y = logits.argmax(-1)
        self.sync_gpu_and_update_metrics(nll_loss.detach(),
                                         soft_dice.detach(),
                                         total_loss.detach(),
                                         pred_y.detach(),
                                         target.detach(), B)

        self.sample_counter += B

        return total_loss