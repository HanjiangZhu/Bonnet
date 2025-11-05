import torch
from typing import Tuple, Dict
import numpy as np

def torch_rotate_points_90deg(voxel_coords: torch.Tensor, supervoxel_coords: torch.Tensor, axis: int, plus: bool) -> torch.Tensor:
    assert len(voxel_coords.shape) == 2 and voxel_coords.shape[-1] == 3
    assert len(supervoxel_coords.shape) == 2 and supervoxel_coords.shape[-1] == 3
    
    axis_swap = [0, 1, 2]
    axis_mult = [1, 1, 1]
    if axis == 0:
        axis_swap = [0, 2, 1]
        axis_mult = [1, -1, 1]
    elif axis == 1:
        axis_swap = [2, 1, 0]
        axis_mult = [1, 1, -1]
    elif axis == 2:
        axis_swap = [1, 0, 2]
        axis_mult = [-1, 1, 1]
    else:
        raise ValueError("Invalid axis value")

    if not plus:
        for i in range(3):
            axis_mult[i] *= -1
    
    voxel_coords = voxel_coords[:, axis_swap]
    voxel_coords *= torch.tensor(axis_mult, dtype=voxel_coords.dtype, device=voxel_coords.device)

    supervoxel_coords = supervoxel_coords[:, axis_swap]
    supervoxel_coords *= torch.tensor(axis_mult, dtype=supervoxel_coords.dtype, device=supervoxel_coords.device)

    return voxel_coords, supervoxel_coords    

def torch_select_block_to_delete(voxel_coords: torch.Tensor, min_block_size: float, max_block_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    sample_size = voxel_coords.max(dim=0).values - voxel_coords.min(dim=0).values
    block_size_min = min_block_size * sample_size
    block_size_max = max_block_size * sample_size
    block_size = torch.rand(3) * (block_size_max - block_size_min) + block_size_min
    block_pos = voxel_coords.float().mean(0).int() + (torch.rand(3) * sample_size / 2.0) - block_size/2
    return block_pos, block_size

def torch_delete_points_in_block(voxel_data: Dict[str, torch.Tensor], block_size: torch.Tensor, block_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Deletes points in a random block of size block_size at position block_pos.
    block_min = block_pos
    block_max = block_pos + block_size
    """
    assert len(voxel_data['voxel_coords'].shape) == 2 and voxel_data['voxel_coords'].shape[-1] == 3
    idxs = torch_get_points_outside_block(voxel_data['voxel_coords'], block_pos, block_pos + block_size)
    for k, v in voxel_data.items():
        voxel_data[k] = v[idxs]
    return voxel_data

def torch_feature_swap(features, max_feature_swap_rate) -> torch.Tensor:
    feature_swap_rate = np.random.uniform(0, max_feature_swap_rate)
    num_swaps = max(int(features.shape[0] * feature_swap_rate), 1)
    perm1 = torch.randperm(features.shape[0])
    perm2 = torch.randperm(features.shape[0])
    features[perm1[:num_swaps]] = features[perm2[:num_swaps]]
    return features

def torch_feature_dropout(features, max_feature_dropout_rate, dropout_value_range: tuple) -> torch.Tensor:
    """
    Randomly dropouts features of shape (N, C) with a dropout rate of max_feature_dropout_rate.
    The replacement values are in range dropout_value_range.
    """
    assert len(features.shape) == 2
    N = features.size(0)
    C = features.size(1)
    feature_dropout_rate = np.random.uniform(0, max_feature_dropout_rate)
    dropout_idxs = torch.randperm(N * C)[:int(float(N * C) * feature_dropout_rate)]

    features = features.view(-1)
    r = dropout_value_range
    features[dropout_idxs] = torch.rand(dropout_idxs.shape[0]) * (r[1] - r[0]) + r[0]
    features = features.view(N, C)
    return features
    

def torch_normalize_feats(voxels_list: Dict[str, torch.Tensor], center: torch.Tensor, scale: torch.Tensor):
    for voxels in voxels_list.items():
        if len(voxels['features'].shape) == 1:
            voxels['features'] = voxels['features'][:, None].float()

        voxels['features'] = (voxels['features'] - center.float()) / scale.float()

@torch.no_grad()
def torch_get_points_in_block(x_data, block_min, block_max):
    assert x_data.shape[-1] == 3
    bool_mask = torch.where((x_data[..., 0] >= block_min[0]) &
                            (x_data[..., 0] < block_max[0]) &
                            (x_data[..., 1] >= block_min[1]) &
                            (x_data[..., 1] < block_max[1]) &
                            (x_data[..., 2] >= block_min[2]) &
                            (x_data[..., 2] < block_max[2]))[0]
    return bool_mask

@torch.no_grad()
def torch_get_points_outside_block(x_data, block_min, block_max):
    assert x_data.shape[-1] == 3
    bool_mask = torch.where((x_data[..., 0] < block_min[0]) |
                            (x_data[..., 0] >= block_max[0]) |
                            (x_data[..., 1] < block_min[1]) |
                            (x_data[..., 1] >= block_max[1]) |
                            (x_data[..., 2] < block_min[2]) |
                            (x_data[..., 2] >= block_max[2]))[0]
    return bool_mask

@torch.no_grad()
def torch_flat_coords_cut_window(voxel_coords: torch.Tensor, voxel_feats: torch.Tensor, voxel_labels: torch.Tensor,
                                 window_size: torch.Tensor, window_pos: torch.Tensor, return_idxs=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(voxel_coords.shape) == 2 and voxel_coords.shape[-1] == 3
    idxs = torch_get_points_in_block(voxel_coords, window_pos, window_pos + window_size)
    voxel_coords = voxel_coords[idxs]
    voxel_feats = voxel_feats[idxs]
    if voxel_labels is not None:
        voxel_labels = voxel_labels[idxs]
    if not return_idxs:
        return voxel_coords, voxel_feats, voxel_labels
    
    return voxel_coords, voxel_feats, voxel_labels, idxs

@torch.no_grad()
def torch_downsample_points_if_necessary(max_pnt_count: int, points_data: Dict[str, torch.Tensor]):
    if not max_pnt_count or max_pnt_count <= 0:
        return

    ref_key = None
    candidate_keys = ("points", "coords", "xyz", "positions", "feats", "features", "labels")
    for k in candidate_keys:
        if k in points_data and torch.is_tensor(points_data[k]) and points_data[k].dim() > 0:
            ref_key = k
            break
    if ref_key is None:
        tensors = [(k, v) for k, v in points_data.items()
                   if torch.is_tensor(v) and v.dim() > 0]
        if not tensors:
            return
        ref_key = max(tensors, key=lambda kv: kv[1].shape[0])[0]

    ref = points_data[ref_key]
    N = ref.shape[0]
    if N <= max_pnt_count:
        return

    idx = torch.randperm(N, device=ref.device)[:max_pnt_count]

    for k, v in list(points_data.items()):
        if not torch.is_tensor(v) or v.dim() == 0:
            continue
        if v.shape[0] == N:
            points_data[k] = v.index_select(0, idx.to(v.device))
        else:
            pass



### TORCH PART - Augmentations

# Always assumes that voxel_coords has shape (N, 3)

def _validate_voxel_coords(voxel_coords: torch.Tensor):
    assert len(voxel_coords.shape) == 2, f"voxel_coords has incorrect shape: {voxel_coords.shape}"
    assert voxel_coords.shape[1] == 3, f"voxel_coords has incorrect shape: {voxel_coords.shape}"

@torch.no_grad()
def torch_calculate_positive_shift(voxel_coords: torch.Tensor, window_size: torch.Tensor) -> torch.Tensor:
    """
    Assumes voxel is shifted to (0,0,0).
    Returns the shift to apply to the voxel to ensure that the voxel is within the window.
    """
    _validate_voxel_coords(voxel_coords)
    max_shift = (window_size - voxel_coords.max(dim=0).values - 1).detach().cpu().numpy()
    shift = [
        np.random.randint(0, max_shift[0]) if max_shift[0] > 0 else 0,
        np.random.randint(0, max_shift[1]) if max_shift[1] > 0 else 0,
        np.random.randint(0, max_shift[2]) if max_shift[2] > 0 else 0
    ]
    shift = torch.tensor(shift, dtype=torch.int32, device=voxel_coords.device)[None, :]
    return shift