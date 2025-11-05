import numpy as np
from typing import Tuple, List, Dict

### NUMPY PART - Cutting windows and downsampling points

def create_supervoxel(voxels: np.array, R: int) -> np.array:
    # Creates supervoxel
    supervoxel = np.zeros((voxels.shape[0]//R, voxels.shape[1]//R, voxels.shape[2]//R, R**3), dtype=np.uint8)
    
    for i in range(R):
        for j in range(R):
            for k in range(R):
                # Select every Rth voxel
                x = voxels[i::R, j::R, k::R]
                # Ensure shape matches result.shape
                x = x[:supervoxel.shape[0], :supervoxel.shape[1], :supervoxel.shape[2]]
                supervoxel[:, :, :, i + j*R + k*R*R] = x

    return supervoxel

def create_mask_and_repeat_points_if_necessary(target_count: int, points_data: Dict[str, np.ndarray], mask_name='voxel_mask'):
    """
    For each point cloud in points_data, repeat the points until the target_count is reached.
    Also creates a mask that is 1 for the points that were actually present in the original point cloud, and 0 for the rest.
    """
    points_count = next(iter(points_data.values())).shape[0]
    ret_mask = np.zeros(target_count, dtype=np.float32)
    ret_mask[:min(target_count, points_count)] = 1

    for key, points in points_data.items():
        if points_count >= target_count:
            points_data[key] = points[:target_count]
        else:
            points_data[key] = np.concatenate([points, points[:target_count - points_count]])
    
    assert mask_name not in points_data, f"Mask name {mask_name} already present in points_data"
    points_data[mask_name] = ret_mask

