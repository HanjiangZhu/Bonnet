import numpy as np
from omegaconf import OmegaConf
import json
import os
from pathlib import Path
import nibabel as nib
import logging
from typing import Tuple
import torch

### Dataset utils

@torch.no_grad()
def collate_coords_flat(coords_list):
    """
    Collates a list of coordinates into a batch.
    """
    coords_list = coords_list
    repeats_count = [len(x) for x in coords_list]
    for i in repeats_count: assert i > 0

    batch_idxs = torch.arange(len(coords_list), dtype=torch.int32)
    batch_idxs = torch.repeat_interleave(batch_idxs, torch.tensor(repeats_count, dtype=batch_idxs.dtype))
    coords = torch.concatenate([x for x in coords_list], axis=0)
    coords = torch.concatenate([batch_idxs[:, None], coords], axis=1)
    return coords

@torch.no_grad()
def collate_flat(samples_list, attrib_name='voxel'):
    """
    Collates a list of samples (dicts) into a batch.
    Flat means (N, 4) shape.
    """
    result = {
        f'{attrib_name}_coords': collate_coords_flat([ x[f'{attrib_name}_coords'] for x in samples_list ]),
        f'{attrib_name}_features': torch.concatenate([x[f'{attrib_name}_features'] for x in samples_list], axis=0),
    }
    if f'{attrib_name}_labels' in samples_list[0]:
        result[f'{attrib_name}_labels'] = torch.concatenate([x[f'{attrib_name}_labels'] for x in samples_list], axis=0)
    return result

# s0000/
    # ct.nii.gz -> HU values (densidades)
    # 116
    # aorta.nii.gz -> Segmentation masks 0-1
    # liver.nii.gz -> Segmentation masks 0-1
    # rib_c1.nii.gz -> Segmentation masks 0-1
    # ...


### Loading

def load_ct_and_flatten_labels(sample_path: str, class_encoder_map: dict, skip_hu_values_load: bool = False, seg_dir: str = "segmentations"):
    try:
        segpath = Path(sample_path)
        voxels = None
        if not skip_hu_values_load:
            voxels = nib.load(segpath.joinpath('ct.nii.gz')).get_fdata()
        labels = None

        if seg_dir is not None:
            segpath = Path(segpath, seg_dir)
        # Load labels
        for cls_name in class_encoder_map.keys():
            cls_idx = class_encoder_map[cls_name.split('.')[0]]
            if cls_idx == 0:
                assert cls_name == '00_other'
                continue
            
            seg = nib.load((segpath / (cls_name + '.nii.gz'))).get_fdata()
            if labels is None:
                labels = np.zeros_like(seg, dtype=np.int32)
            seg = (seg > 0.5).astype(np.int32) * cls_idx
            conflict_mask = (seg > 0) & (labels > 0)

            # Count conflict points
            num_conflicts = np.count_nonzero(conflict_mask)

            perc_conflicts = num_conflicts / conflict_mask.size
            assert perc_conflicts <= 0.05, f"Conflict percentage too high: {perc_conflicts}"

            if num_conflicts > 0:
                # To resolve the conflict set all conflict points to 0
                labels[conflict_mask] = 0
            
            labels += seg
    except Exception as e:
        logging.info(f"Error loading {sample_path}: {e}")
        return None, None

    return voxels, labels

def voxel_to_cloud(data_voxel: np.ndarray, label_voxel: np.ndarray, hu_min: int, hu_max: int):
    # Hu values: 200 e 3000 -> ossos
    data_voxel_mask = (data_voxel >= hu_min) & (data_voxel <= hu_max)

    indices = np.argwhere(data_voxel_mask)
    hu_values = data_voxel[indices[:, 0], indices[:, 1], indices[:, 2]]
    pcloud = indices.astype(np.float32)
    labels = None
    if label_voxel is not None:
        labels = label_voxel[indices[:, 0], indices[:, 1], indices[:, 2]]
    return pcloud, labels, hu_values



def voxel_to_padded_supervoxel(input_voxel: np.ndarray, R: int):
    W = ((input_voxel.shape[0] + (R-1)) // R) * R
    H = ((input_voxel.shape[1] + (R-1)) // R) * R
    D = ((input_voxel.shape[2] + (R-1)) // R) * R

    supervoxel = np.zeros((W, H, D), dtype=input_voxel.dtype)
    supervoxel[:input_voxel.shape[0], :input_voxel.shape[1], :input_voxel.shape[2]] = input_voxel
    
    supervoxel = supervoxel.reshape(W//R, R, H//R, R, D//R, R)
    supervoxel = supervoxel.transpose(0, 2, 4, 1, 3, 5)
    supervoxel = supervoxel.reshape(W//R, H//R, D//R, R**3)
    return supervoxel

def voxel_to_superpoints(hu_voxels: np.ndarray, hu_min: float, hu_max: float, R: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns supervoxel coordinates and features.
    ! NOTE: The returned coordinates are not normalized (i.e. they are in voxel space, not in supervoxel space).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: supervoxel_coords, supervoxel_features
    """
    
    supervoxel = voxel_to_padded_supervoxel(hu_voxels, R)
    simplified_super = np.any((supervoxel >= hu_min) & (supervoxel <= hu_max), axis=-1)
    indices = np.argwhere(simplified_super).astype(np.int32)
    
    supervoxel_coords = indices * int(R)
    supervoxel_coords = supervoxel_coords.astype(np.int32)
    supervoxel_features = supervoxel[indices[:, 0], indices[:, 1], indices[:, 2], :]
    supervoxel_features = supervoxel_features.astype(np.float32)
    return supervoxel_coords, supervoxel_features


### Splits, classes and cache utils


def get_class_encoder_map(name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load("../conf/classes.yaml")[name])

def dict_to_filename(data):
    # Serialize dictionary to JSON string
    json_string = json.dumps(data, sort_keys=True, separators=(',', ':')).replace('\n', '').replace(':', '=').replace('\"', '').replace('[', '(').replace(']', ')').replace('{', '').replace('}', '')
    return json_string

def get_ct_numbers(dataset_path):
    ct_nums = sorted([ int(x[1:].split(".")[0]) for x in os.listdir(dataset_path) if x.startswith("s") ])
    assert int(len(ct_nums)) == 1228, "Expected 1228 CT scans, got {}".format(len(ct_nums))
    return np.asarray(ct_nums)

def train_val_split(train_idxs):
    val_size = 0.125
    return train_idxs[:int(len(train_idxs)*(1 - val_size))], train_idxs[int(len(train_idxs)*(1 - val_size)):]

def ids_to_ct_idxs(ids, ct_numbers):
    # ids = sXXXX
    # ct_numbers = [XXXX, YYYY, ...]
    return np.where(np.isin(ct_numbers, [int(x[1:].split(".")[0]) for x in ids]))[0]

def validate_splits(meta, idxs, split, ct_numbers):
    for idx in idxs:
        # Find the row in meta with image_id == sXXXX
        row = meta[meta["image_id"] == "s{}".format(str(ct_numbers[idx]).zfill(4))]
        assert row["split"].values[0] == split, "Invalid split for {}".format(ct_numbers[idx])

def create_splits(ct_numbers, fold_idx):
    import numpy as np
    import pandas as pd

    with open("/mnt/sdb/Bonnet-master/splits_final.json") as f:
        splits_final = json.load(f)

    df = pd.read_csv("/mnt/sdb/Bonnet-master/DS_ORIGINAL/meta.csv", sep=";")

    assert fold_idx == 0, "Only fold 0 is supported for now"

    train_ids = splits_final[fold_idx]["train"]
    val_ids = splits_final[fold_idx]["val"]
    test_ids = df[df["split"] == "test"]["image_id"].values

    train_idxs = ids_to_ct_idxs(train_ids, ct_numbers)
    val_idxs = ids_to_ct_idxs(val_ids, ct_numbers)
    test_idxs = ids_to_ct_idxs(test_ids, ct_numbers)

    # Assert that there is no overlap between the splits
    assert len(np.intersect1d(train_idxs, val_idxs)) == 0, "Overlap between train and val"
    assert len(np.intersect1d(train_idxs, test_idxs)) == 0, "Overlap between train and test"
    assert len(np.intersect1d(val_idxs, test_idxs)) == 0, "Overlap between val and test"

    assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(ct_numbers), "Invalid split sizes"
    return np.array(train_idxs, dtype=int), np.array(val_idxs, dtype=int), np.array(test_idxs, dtype=int)

