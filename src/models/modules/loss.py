import torch.nn as nn

import models.modules.functional as modules_F

__all__ = ['KLLoss']


class KLLoss(nn.Module):
    def forward(self, x, y):
        return modules_F.kl_loss(x, y)
