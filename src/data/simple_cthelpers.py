import numpy as np
import torch
from .voxels import Voxels, VoxelsFormat
from typing import Tuple, List
import math
from typing import Optional

# dataset characteristics according to nnunet
GENERAL_PERCENTILE_00_5 = -1003.0
GENERAL_PERCENTILE_99_5 = 1546.0
# GENERAL_STD = 505.3545227050781
# GENERAL_MEAN = -104.43730926513672
GENERAL_MEAN = 471.46878
GENERAL_STD = 286.46423
GENERAL_MAX = 11368.0
GENERAL_MEDIAN = 39.0
GENERAL_MIN = -9052.0
# 471.46878
# 286.46423
# 470.20337
# 285.51396

def get_ct_fullname(ct_number: int):
    return f's{str(ct_number).zfill(4)}'

@torch.no_grad()
def get_bone_coords(ct: np.ndarray):
    return torch.argwhere((ct >= 200) & (ct <= 3000)).int()

@torch.no_grad()
def normalize_hus(ct: torch.Tensor, mean=GENERAL_MEAN, std=GENERAL_STD):
    ct = torch.clip(ct.float(), GENERAL_MIN, GENERAL_MAX)
    ct = (ct - mean) / std
    return ct

@torch.no_grad()
def crop_coords_in_inference_window(idx, dim, overlap_multiplier, window_size, dimMin, dimMax, voxel_coords):
    box_dim_min = idx * window_size[dim].item() * overlap_multiplier + dimMin
    box_dim_max = box_dim_min + window_size[dim].item()

    if box_dim_max > dimMax: # Clip last patch
        box_dim_max = dimMax
        box_dim_min = box_dim_max - window_size[dim].item()
    
    coords_in_box = voxel_coords[(voxel_coords[:, dim] >= box_dim_min) & (voxel_coords[:, dim] < box_dim_max)]
    return coords_in_box

def get_num_patches_for_inference(dimMin, dimMax, window_size, dim, test_window_overlap):
    return int(math.ceil((dimMax - dimMin) / window_size[dim] / test_window_overlap))

def get_occ_coords(occ_hus: torch.Tensor, occ_labels: torch.Tensor):
    occ_mask_to_sample = torch.ones_like(occ_hus)
    occ_coords = torch.argwhere(occ_mask_to_sample).int()
    return occ_coords

@torch.no_grad()
def get_all_windows(bones: Voxels, occ: Optional[Voxels],
                    window_size: torch.Tensor) -> List[Tuple[Voxels, Voxels]]:
    result = []
    test_window_overlap = 0.5
    overlap_multiplier = 1.0 - test_window_overlap

    yMin, yMax = bones.coords[:, 1].min().item(), bones.coords[:, 1].max().item()
    num_height_patches = get_num_patches_for_inference(yMin, yMax, window_size, 1, test_window_overlap)

    for h in range(num_height_patches):
        coords_in_yBox = crop_coords_in_inference_window(h, 1, overlap_multiplier, window_size, yMin, yMax, bones.coords)
        xMin, xMax = coords_in_yBox[:, 0].min().item(), coords_in_yBox[:, 0].max().item()
        num_width_patches = get_num_patches_for_inference(xMin, xMax, window_size, 0, test_window_overlap)

        for w in range(num_width_patches):
            coords_in_xyBox = crop_coords_in_inference_window(w, 0, overlap_multiplier, window_size, xMin, xMax, coords_in_yBox)
            zMin, zMax = coords_in_xyBox[:, 2].min().item(), coords_in_xyBox[:, 2].max().item()
            num_depth_patches = get_num_patches_for_inference(zMin, zMax, window_size, 2, test_window_overlap)

            for d in range(num_depth_patches):
                window_pos = torch.tensor([
                    w * window_size[0].item() * overlap_multiplier + xMin,
                    h * window_size[1].item() * overlap_multiplier + yMin,
                    d * window_size[2].item() * overlap_multiplier + zMin
                ], dtype=torch.int32)

                if window_pos[0] + window_size[0] > xMax: window_pos[0] = xMax - window_size[0]
                if window_pos[1] + window_size[1] > yMax: window_pos[1] = yMax - window_size[1]
                if window_pos[2] + window_size[2] > zMax: window_pos[2] = zMax - window_size[2]

                bones_window, bones_window_indices = bones.crop_window(window_pos, window_size)
                occ_window, occ_window_indices = occ.crop_window(window_pos, window_size) if occ is not None else (None, None)

                result.append({
                    'bones': bones_window,
                    'occ': occ_window,
                    'window_pos': window_pos,
                    'bones_window_indices': bones_window_indices,
                    'occ_window_indices': occ_window_indices
                })
    
    return result