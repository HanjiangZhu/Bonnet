import numpy as np
import torch
from typing import Union
from metrics.base_metric import BaseMetric

# A metric that aggregates values and takes the rolling mean
class AggMean(BaseMetric):
    def __init__(self):
        self.mean = 0.0
        self.value_count = 0
    
    def reset(self):
        self.mean = 0.0
        self.value_count = 0

    @torch.no_grad()
    def update(self, value: Union[np.ndarray, torch.tensor], count: int):
        assert torch.is_tensor(value)
        value = value.detach()

        self.mean = (self.mean * self.value_count + value * count) / (self.value_count + count)
        self.value_count += count

    def compute(self) -> np.ndarray:
        if torch.is_tensor(self.mean):
            return self.mean.item()
        return self.mean

    def as_wandb_metric(self) -> np.ndarray:
        if torch.is_tensor(self.mean):
            return self.mean.item()
        return self.mean