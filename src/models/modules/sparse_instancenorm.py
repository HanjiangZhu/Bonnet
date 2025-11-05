import torch
import torch.nn as nn
from ..backends.utils.spconv_utils import spconv, replace_feature

class SparseInstanceNorm(spconv.SparseModule):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine=True):
        super().__init__()
        assert affine, "Currently only supports affine=True"
        self.eps = eps
        self.affine = affine
        self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32, requires_grad=True))
    
    def forward(self, x: spconv.SparseConvTensor):
        """
        x has feats: (N, C), coords: (N, 4)

        where 4 = batch_idx, x, y, z
        """
        final_feats = []
        for b in range(x.batch_size):
            mask = x.indices[:, 0] == b
            # Assert mask is contiguous
            # assert ind.min() == cnt and ind.max() == cnt + ind.shape[0] - 1 and ind.shape[0] == ind.max() - ind.min() + 1
            feats = x.features[mask]
            mean = feats.mean(dim=0)
            var = feats.var(dim=0, unbiased=True)
            new_feats = (feats - mean) / (var + self.eps).sqrt() * self.weight.to(feats.dtype) + self.bias.to(feats.dtype)
            final_feats.append(new_feats)
        
        final_feats = torch.cat(final_feats, dim=0)
        x = replace_feature(x, final_feats)
        return x