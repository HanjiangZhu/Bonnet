import torch
import torch.nn as nn
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Conv3d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def gaussian_kernel(coords, voxel_size, std_scale=1.0):
    # assert len(coords.shape) == 2
    # Applies a gaussian kernel based on the input coords
    center = voxel_size.float() / 2
    dists = (coords[..., :3].float() - center) / center
    dists = torch.sum(dists * dists, dim=-1)
    # Apply gaussian pdf
    inv_std = 1.0 / std_scale
    kernel = torch.exp(-dists * 0.5 * inv_std**2)
    return kernel