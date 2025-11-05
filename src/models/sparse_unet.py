import torch
import torch.nn as nn
from models.backends.spconv_unet import UNetV2
from models.models_utils import weights_init

__all__ = ['SparseUNet']


class SparseUNet(nn.Module):
    def __init__(self, num_classes: int, window_size: list, width: float = 2.0):
        super(SparseUNet, self).__init__()

        voxel_size = 1
        point_cloud_range = [0, 0, 0]

        self.num_classes = num_classes
        self.window_size = window_size
        self.backbone = UNetV2(1, window_size, voxel_size, point_cloud_range, width=width)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_point_features, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes)
        )
        self.apply(weights_init)
    
    def forward(self, x, collect_stats=False):
        # assert torch.is_tensor(x['voxel_coords']), "Input voxel_coords is not a tensor. Type: {}".format(type(x['voxel_coords']))
        # assert torch.all(x['voxel_coords'][:, 1:].max(dim=0).values < torch.tensor(self.window_size, dtype=x['voxel_coords'].dtype, device=x['voxel_coords'].device)), "Input voxel_coords are out of bounds!"
        # assert torch.all(x['voxel_coords'][:, 1:].min(dim=0).values >= 0), "Input voxel_coords are out of bounds!"
        # assert torch.unique(x['voxel_coords'][:, 0]).shape[0] == x['batch_size'], "Batch size is inconsistent with voxel_coords!"
        # Also check that batch_size is consistent
        ret = self.backbone(x['voxel_coords'], x['voxel_features'], x['batch_size'], collect_stats=collect_stats)
        # NOTE: ret.indices is the reversed indices!!!!
        # This returns true assert torch.all(ret.indices[:, [3,2,1]] == x['voxel_coords'][..., 1:]), "Indices are not the same!"
        ret = ret.features
        if collect_stats: return ret
        logits = self.classifier(ret)
        return logits
