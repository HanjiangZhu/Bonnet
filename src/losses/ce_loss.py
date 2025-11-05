import torch
import torch.nn.functional as F
from typing import Union
from omegaconf import DictConfig
from metrics.agg_cfmat import AggCfMatrix
from metrics.agg_mean import AggMean
from losses.base_loss import BaseLoss
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class CELoss(BaseLoss):
    def __init__(self, cfg: DictConfig, label_smoothing: float):
        super(CELoss, self).__init__()
        self.cfg = cfg
        self.metrics = self.create_metrics()
        self.sample_counter = 0
        self.label_smoothing = label_smoothing

    def create_metrics(self):
        metrics = {}
        metrics['cfmat'] = AggCfMatrix(self.cfg.data.class_names)
        metrics['acc'] = AggMean()
        metrics['totalLoss'] = AggMean()
        metrics['f1'] = AggMean()
        metrics['precision'] = AggMean()
        metrics['recall'] = AggMean()
        return metrics

    def on_start_epoch(self):
        super().on_start_epoch()
        self.sample_counter = 0
        self.all_preds = []
        self.all_targets = []
    
    def on_end_epoch(self):
        super().on_end_epoch()

        target = np.concatenate(self.all_targets)
        pred_y = np.concatenate(self.all_preds)

        precision = precision_score(target, pred_y, average='macro')
        recall = recall_score(target, pred_y, average='macro')
        f1 = f1_score(target, pred_y, average='macro')

        self.metrics['f1'].update(f1, 1)
        self.metrics['precision'].update(precision, 1)
        self.metrics['recall'].update(recall, 1)
    
    def __call__(self,
                 logits: torch.FloatTensor,
                 target: Union[torch.LongTensor, torch.FloatTensor]):
        nll_loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        total_loss = nll_loss
        B = target.size(0)

        pred_y = logits.argmax(-1).cpu().detach().numpy()
        target_y = target.long().cpu().detach().numpy()
        acc = np.mean(pred_y == target_y)
        self.metrics['acc'].update(acc, B)
        self.metrics['cfmat'].update(pred_y, target_y)
        self.metrics['totalLoss'].update(total_loss.item(), B)

        self.all_preds.append(pred_y)
        self.all_targets.append(target_y)
        return total_loss