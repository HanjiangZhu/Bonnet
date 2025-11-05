import numpy as np
import sklearn.metrics as skm
from typing import List, Union
import metrics.wandbutils as wandbutils
from metrics.base_metric import BaseMetric
import torch
import wandb

@torch.no_grad()
def create_labels_lookup_table(num_classes, labels):
    """
    Creates a lookup table for labels of size num_classes.
    Maps all classes to a continuous range of [0, len(labels)].
    """

    labels, _ = torch.sort(labels)
    labels_lookup_table = torch.zeros(num_classes, dtype=torch.int64, device=labels.device) - 1
    labels_lookup_table[labels] = torch.arange(len(labels), dtype=torch.int64, device=labels.device)
    return labels_lookup_table, labels

@torch.no_grad()
def confusion_matrix_vectorized(y_true, y_pred, num_classes):
    """
    Compute the confusion matrix using a vectorized approach with PyTorch.
    
    Parameters:
    y_true (torch.Tensor): Tensor of true labels.
    y_pred (torch.Tensor): Tensor of predicted labels.
    num_classes (int): Number of classes.
    
    Returns:
    torch.Tensor: Confusion matrix.
    """
    # Initialize the confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=y_true.device)

    # Use torch's advanced indexing to accumulate confusion matrix values
    indices = num_classes * y_true + y_pred
    conf_matrix.put_(indices, torch.ones_like(indices, device=y_true.device), accumulate=True)

    return conf_matrix

@torch.no_grad()
def dice_from_cfmat(cfmat, include_bg=False):
    """
    Compute the Dice score from a confusion matrix.
    """

    dice = (2.0 * torch.diag(cfmat) + 1e-8) / (torch.sum(cfmat, 1) + torch.sum(cfmat, 0) + 1e-8)
    if not include_bg:
        dice = dice[1:]
    return torch.mean(dice)

class AggCfMatrix(BaseMetric):
    # Computes global dice with GPU support (to avoid syncing to CPU)

    def __init__(self,
                 class_names: List[str],
                 device: torch.device,
                 labels: torch.Tensor = None,
                 normalize_total: bool = False,
                 log_cfmat: bool = False):
        self.device = device
    
        self.labels = None
        self.labels_lookup_table = None
    
        self.class_names = class_names
        self.normalize_total = normalize_total
        self.log_cfmat = log_cfmat

        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.int64) if not torch.is_tensor(labels) else labels.clone().detach()
            assert labels.max() < len(class_names), "Labels must be in range [0, num_classes)"
            assert labels.min() >= 0, "Labels must be in range [0, num_classes)"
            self.labels_lookup_table, self.labels = create_labels_lookup_table(self.num_classes, labels)
            # NOTE: self.labels is the sorted version of labels. Either way you should pass in sorted labels...
            self.class_names = [class_names[i.item()] for i in self.labels]
            assert self.num_classes == len(self.labels), "Number of classes must match the number of labels"

        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=device)

    @property
    def num_classes(self):
        return len(self.class_names)

    def reset(self):
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None):
        """
        Aggregates y_pred and y_true into a confusion matrix, according to the mask.
        NOTE: y_pred should be provided according to the labels subset provided in the constructor.

        Args:
            y_pred (torch.Tensor): A tensor of shape S in range [0, num_classes)
            y_true (torch.Tensor): A tensor of shape S in range [0, num_classes)
            mask (torch.Tensor): A tensor of shape S in range [0, 1)
        """
        assert torch.is_tensor(y_pred) and torch.is_tensor(y_true) and y_pred.shape == y_true.shape
        if mask is not None:
            assert torch.is_tensor(mask) and mask.shape == y_pred.shape
            assert mask.dtype == torch.bool, "Mask must be of type bool"
            y_pred = y_pred[mask]
            y_true = y_true[mask]

        mapped_y_pred = y_pred
        mapped_y_true = y_true
        
        if self.labels_lookup_table is not None:
            mapped_y_pred = self.labels_lookup_table[y_pred]
            mapped_y_true = self.labels_lookup_table[y_true]
            assert torch.all(mapped_y_pred >= 0) and torch.all(mapped_y_true >= 0), "Labels must be in range [0, num_classes)! Passed an invalid prediction or target label."

        cfmat = confusion_matrix_vectorized(mapped_y_true, mapped_y_pred, self.num_classes)
        self.matrix += cfmat

    @torch.no_grad()
    def compute(self) -> np.ndarray:
        """Normalizes and returns the confusion matrix

        Returns:
            np.npdarray: The NORMALIZED confusion matrix.
        """
        matrix = self.matrix.detach().cpu().numpy().astype(np.float32)
        if self.normalize_total:
            matrix /= np.sum(matrix) + 1e-8
        else:
            matrix /= (np.sum(matrix, axis=1, keepdims=True) + 1e-8)
        return matrix
    
    @torch.no_grad()
    def dice(self, include_bg: bool) -> np.ndarray:
        # compute dice from confusion matrix
        dice = dice_from_cfmat(self.matrix, include_bg)
        return dice.item()
    
    def as_wandb_metric(self) -> Union[np.ndarray, wandb.Image]:
        if self.log_cfmat:
            return wandbutils.create_conf_matrix(self.class_names, self.compute())
        else:
            return self.dice(include_bg=False)