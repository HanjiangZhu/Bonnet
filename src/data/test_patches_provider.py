import math
import torch
from data.torch_datatransform import torch_flat_coords_cut_window

def get_coords_in_bounds(idx, dim, overlap_multiplier, window_size, dimMin, dimMax, voxel_coords):
    box_dim_min = idx * window_size[dim].item() * overlap_multiplier + dimMin
    box_dim_max = box_dim_min + window_size[dim].item()

    if box_dim_max > dimMax: # Clip last patch
        box_dim_max = dimMax
        box_dim_min = box_dim_max - window_size[dim].item()
    
    coords_in_box = voxel_coords[(voxel_coords[:, dim] >= box_dim_min) & (voxel_coords[:, dim] < box_dim_max)]
    return coords_in_box

def get_num_patches(dimMin, dimMax, window_size, dim, test_window_overlap):
    return int(math.ceil((dimMax - dimMin) / window_size[dim] / test_window_overlap))

def get_all_patches(voxel_coords, voxel_features, voxel_labels, window_size, test_window_overlap):
    all_samples = []
    overlap_multiplier = 1.0 - test_window_overlap

    yMin, yMax = voxel_coords[:, 1].min().item(), voxel_coords[:, 1].max().item()
    num_height_patches = get_num_patches(yMin, yMax, window_size, 1, test_window_overlap)

    for h in range(num_height_patches):
        coords_in_yBox = get_coords_in_bounds(h, 1, overlap_multiplier, window_size, yMin, yMax, voxel_coords)
        xMin, xMax = coords_in_yBox[:, 0].min().item(), coords_in_yBox[:, 0].max().item()
        num_width_patches = get_num_patches(xMin, xMax, window_size, 0, test_window_overlap)

        for w in range(num_width_patches):
            coords_in_xyBox = get_coords_in_bounds(w, 0, overlap_multiplier, window_size, xMin, xMax, coords_in_yBox)
            zMin, zMax = coords_in_xyBox[:, 2].min().item(), coords_in_xyBox[:, 2].max().item()
            num_depth_patches = get_num_patches(zMin, zMax, window_size, 2, test_window_overlap)

            for d in range(num_depth_patches):
                lbs = torch.tensor([
                    w * window_size[0].item() * overlap_multiplier + xMin,
                    h * window_size[1].item() * overlap_multiplier + yMin,
                    d * window_size[2].item() * overlap_multiplier + zMin
                ], dtype=torch.int32)

                if lbs[0] + window_size[0] > xMax:
                    lbs[0] = xMax - window_size[0]
                
                if lbs[1] + window_size[1] > yMax:
                    lbs[1] = yMax - window_size[1]

                if lbs[2] + window_size[2] > zMax:
                    lbs[2] = zMax - window_size[2]
                
                voxel_coords_window, voxel_features_window, voxel_labels_window, window_indices = torch_flat_coords_cut_window(voxel_coords, voxel_features, voxel_labels, window_size, lbs, True)

                if voxel_coords_window.shape[0] < 100:
                    continue
                
                sample = {
                    'voxel_coords': voxel_coords_window,
                    'voxel_features': voxel_features_window,
                    'voxel_labels': voxel_labels_window,
                    'window_indices': window_indices
                }
                sample['voxel_coords'] -= sample['voxel_coords'].min(axis=0, keepdims=True).values
                all_samples.append(sample)
    
    return all_samples

def get_all_patches_supervoxel(supervoxel_coords, supervoxel_features, voxel_coords, voxel_features, voxel_labels, window_size, test_window_overlap):
    all_samples = []
    overlap_multiplier = 1.0 - test_window_overlap

    yMin, yMax = voxel_coords[:, 1].min().item(), voxel_coords[:, 1].max().item()
    num_height_patches = get_num_patches(yMin, yMax, window_size, 1, test_window_overlap)

    for h in range(num_height_patches):
        coords_in_yBox = get_coords_in_bounds(h, 1, overlap_multiplier, window_size, yMin, yMax, voxel_coords)
        xMin, xMax = coords_in_yBox[:, 0].min().item(), coords_in_yBox[:, 0].max().item()
        num_width_patches = get_num_patches(xMin, xMax, window_size, 0, test_window_overlap)

        for w in range(num_width_patches):
            coords_in_xyBox = get_coords_in_bounds(w, 0, overlap_multiplier, window_size, xMin, xMax, coords_in_yBox)
            zMin, zMax = coords_in_xyBox[:, 2].min().item(), coords_in_xyBox[:, 2].max().item()
            num_depth_patches = get_num_patches(zMin, zMax, window_size, 2, test_window_overlap)

            for d in range(num_depth_patches):
                lbs = torch.tensor([
                    w * window_size[0].item() * overlap_multiplier + xMin,
                    h * window_size[1].item() * overlap_multiplier + yMin,
                    d * window_size[2].item() * overlap_multiplier + zMin
                ], dtype=torch.int32)

                if lbs[0] + window_size[0] > xMax:
                    lbs[0] = xMax - window_size[0]
                
                if lbs[1] + window_size[1] > yMax:
                    lbs[1] = yMax - window_size[1]

                if lbs[2] + window_size[2] > zMax:
                    lbs[2] = zMax - window_size[2]
                
                supervoxel_coords_window, supervoxel_features_window, _ = torch_flat_coords_cut_window(supervoxel_coords, supervoxel_features, None,
                                                                                                       window_size, lbs, False)
                
                voxel_coords_window, voxel_features_window, voxel_labels_window, window_indices = torch_flat_coords_cut_window(voxel_coords, voxel_features, voxel_labels,
                                                                                                                                window_size, lbs, True)

                if voxel_coords_window.shape[0] < 100:
                    continue
                
                sample = {
                    'supervoxel_coords': supervoxel_coords_window,
                    'supervoxel_features': supervoxel_features_window,
                    'voxel_coords': voxel_coords_window,
                    'voxel_features': voxel_features_window,
                    'voxel_labels': voxel_labels_window,
                    'window_indices': window_indices
                }
                sample['supervoxel_coords'] -= lbs
                sample['voxel_coords'] -= lbs
                all_samples.append(sample)
    
    return all_samples

def get_all_patches_query(voxel_coords, voxel_features, voxel_labels, window_size, test_window_overlap, ctdataset_occupancy, input_batch):
    all_samples = []
    overlap_multiplier = 1.0 - test_window_overlap

    yMin, yMax = voxel_coords[:, 1].min().item(), voxel_coords[:, 1].max().item()
    num_height_patches = get_num_patches(yMin, yMax, window_size, 1, test_window_overlap)

    for h in range(num_height_patches):
        coords_in_yBox = get_coords_in_bounds(h, 1, overlap_multiplier, window_size, yMin, yMax, voxel_coords)
        xMin, xMax = coords_in_yBox[:, 0].min().item(), coords_in_yBox[:, 0].max().item()
        num_width_patches = get_num_patches(xMin, xMax, window_size, 0, test_window_overlap)

        for w in range(num_width_patches):
            coords_in_xyBox = get_coords_in_bounds(w, 0, overlap_multiplier, window_size, xMin, xMax, coords_in_yBox)
            zMin, zMax = coords_in_xyBox[:, 2].min().item(), coords_in_xyBox[:, 2].max().item()
            num_depth_patches = get_num_patches(zMin, zMax, window_size, 2, test_window_overlap)

            for d in range(num_depth_patches):
                lbs = torch.tensor([
                    w * window_size[0].item() * overlap_multiplier + xMin,
                    h * window_size[1].item() * overlap_multiplier + yMin,
                    d * window_size[2].item() * overlap_multiplier + zMin
                ], dtype=torch.int32)

                if lbs[0] + window_size[0] > xMax:
                    lbs[0] = xMax - window_size[0]
                
                if lbs[1] + window_size[1] > yMax:
                    lbs[1] = yMax - window_size[1]

                if lbs[2] + window_size[2] > zMax:
                    lbs[2] = zMax - window_size[2]
               
                voxel_coords_window, voxel_features_window, voxel_labels_window, window_indices = torch_flat_coords_cut_window(voxel_coords, voxel_features, voxel_labels,
                                                                                                                                window_size, lbs, True)

                query_uniform_coords, query_uniform_features, query_uniform_labels = ctdataset_occupancy.generate_uniform_points(input_batch, lbs)

                if voxel_coords_window.shape[0] < 100:
                    continue
                
                sample = {
                    'query_coords': query_uniform_coords,
                    'query_features': query_uniform_features,
                    'query_labels': query_uniform_labels,
                    'voxel_coords': voxel_coords_window,
                    'voxel_features': voxel_features_window,
                    'voxel_labels': voxel_labels_window,
                    'window_indices': window_indices,
                    'window_pos': lbs
                }
                sample['query_coords'] -= lbs
                sample['voxel_coords'] -= lbs
                all_samples.append(sample)
    
    return all_samples