from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
import logging
import numpy as np
import random
import torch
from typing import List
from dataclasses import dataclass
from omegaconf import OmegaConf
import logging
from .dataloader_wrapper import StepsPerEpochDataLoader
import nibabel as nib
from pathlib import Path
from .voxels import Voxels, VoxelsFormat
import torch.nn.functional as F
from .simple_cthelpers import get_ct_fullname, normalize_hus, get_bone_coords, get_all_windows, get_occ_coords

__all__ = ['SecondStageCTDataset']

def seed_worker_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

@dataclass
class SecondStageCTDataset(Dataset):
    """Dataset that deals with .nib CT scans"""

    # basic info
    cfg: DictConfig
    ct_numbers: List[int]
    split: str
    dataset_path: str
    occ_dataset_path: str
    occ_lowres_factor: int
    window_size: List[int]
    
    # data loader
    num_workers: int
    batch_size: int

    # class info
    num_classes: int
    class_encoder_map: DictConfig
    class_colors: List[str]
    class_names: List[str]

    # extra functionality
    include_occupancy: bool
    bones_batch_format: VoxelsFormat = VoxelsFormat.FLAT
    occ_batch_format: VoxelsFormat = VoxelsFormat.BATCHED

    # legacy
    eval_only: bool = False
    stats: dict = None

    def __post_init__(self):
        super().__init__()

        if self.split == 'test':
            assert self.batch_size == 1, "Batch size must be 1 for test split"

        self.dataset_path = Path(self.dataset_path)
        self.occ_dataset_path = Path(self.occ_dataset_path)
        bone_classes = OmegaConf.load("conf/classes.yaml")['bones']
        bone_classes = [k for k, v in sorted(bone_classes.items(), key=lambda item: item[1])]
        self.bones_class_indices = [self.class_names.index(c) for c in bone_classes]
        self.bones_class_indices = torch.tensor(self.bones_class_indices, dtype=torch.int64)
        self.organs_class_indices = [i for i in range(self.num_classes) if i not in self.bones_class_indices]
        self.window_size = torch.tensor(self.window_size).int()

    def get_ct_data_path(self, idx: int, ds_path: Path):
        return ds_path / get_ct_fullname(self.ct_numbers[idx % len(self.ct_numbers)]) / "ct.npz"

    def get_ct_label_path(self, idx: int, ds_path: Path):
        return ds_path / get_ct_fullname(self.ct_numbers[idx % len(self.ct_numbers)]) / "labels.npz"

    def collate_fn(self, batch):
        if self.split == 'train' or self.split == 'val':
            bones_batch = [b['bones'] for b in batch]
            occ_batch = [b['occ'] for b in batch]
            return [{
                'bones': Voxels.collate(bones_batch, self.bones_batch_format),
                'occ': Voxels.collate(occ_batch, self.occ_batch_format),
                'ct_number': torch.cat([b['ct_number'] for b in batch], dim=0),
                'batch_size': len(batch)
            }]
        else:
            # If we are in test mode, collate is not handled here
            return batch[0]
    
    @torch.no_grad()
    def from_disk(self, idx: int, ds_path: Path):
        hus = torch.from_numpy(np.load(self.get_ct_data_path(idx, ds_path))['arr_0']).float()
        labels = torch.from_numpy(np.load(self.get_ct_label_path(idx, ds_path))['arr_0']).long()
        return hus, labels

    @torch.no_grad()
    def __getitem__(self, idx: int):
        bones_hus, bones_labels = self.from_disk(idx, self.dataset_path)
        occ_hus, occ_labels = self.from_disk(idx, self.occ_dataset_path) if self.include_occupancy else (None, None)
        bone_coords = torch.argwhere((bones_hus >= 200) & (bones_hus <= 3000) & torch.isin(bones_labels, self.bones_class_indices) & bones_labels > 0).int()
        bones = Voxels.from_dense(bone_coords, bones_labels, bones_labels)
        bones.feats = F.one_hot(bones.feats-1, num_classes=self.cfg.data.num_classes-1) # encodes GT as one-hot feature input

        occ = Voxels.from_dense(get_occ_coords(occ_hus, occ_labels), occ_hus, occ_labels) if self.include_occupancy else None
        occ.coords *= self.occ_lowres_factor

        num_uniform_bg_points = torch.sum(occ.labels == 0)
        uniform_query_points_max_bg = 0.25
        max_bg = int(occ.coords.shape[0] * uniform_query_points_max_bg)
        if num_uniform_bg_points > max_bg:
            # re-order BG so that they are at the end
            bg_idxs = torch.where(occ.labels == 0)[0]
            fg_idxs = torch.where(occ.labels != 0)[0]
            # remove some BG points
            bg_idxs = bg_idxs[torch.randperm(bg_idxs.shape[0])[:max_bg]]
            occ.coords = torch.cat([occ.coords[fg_idxs], occ.coords[bg_idxs]], dim=0)
            occ.feats = torch.cat([occ.feats[fg_idxs], occ.feats[bg_idxs]], dim=0)
            occ.labels = torch.cat([occ.labels[fg_idxs], occ.labels[bg_idxs]], dim=0)
            if occ.mask is not None:
                occ.mask = torch.cat([occ.mask[fg_idxs], occ.mask[bg_idxs]], dim=0)

        bones.feats = normalize_hus(bones.feats)
        if occ is not None:
            occ.feats = normalize_hus(occ.feats)

        ct_number = torch.tensor([self.ct_numbers[idx % len(self.ct_numbers)]], dtype=torch.int32)

        if self.split == 'train' or self.split == 'val':
            MAX_POINTS = 300000
            bones = bones.ensure_size(MAX_POINTS)
            occ = occ.ensure_size(MAX_POINTS)

            return {
                'bones': bones,
                'occ': occ,
                'ct_number': ct_number
            }
        else:
            ret = get_all_windows(bones, occ, self.window_size)
            for x in ret:
                x['ct_number'] = ct_number
            return ret

    def __len__(self):
        L = len(self.ct_numbers)
        return L

    def to_dataloader(self, shuffle, steps_per_epoch: int, enable_data_prefetch: bool):
        g = torch.Generator()
        g.manual_seed(0)
        logging.info(f"Creating {self.split} dataloader... Batch size: {self.batch_size}")

        ds_loader = DataLoader(self,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         worker_init_fn=seed_worker_fn,
                         pin_memory=True,
                         collate_fn=self.collate_fn,
                         generator=g,
                         shuffle=shuffle
                    )
        
        if self.split == 'train' and steps_per_epoch is not None:
            ds_loader = StepsPerEpochDataLoader(ds_loader, steps_per_epoch)
        
        assert not enable_data_prefetch, "Data prefetching is not supported for SimpleCTDataset"
        # ds_loader = DataLoaderCUDABus(ds_loader, enable_prefetch=enable_data_prefetch)
        return ds_loader