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
from .simple_cthelpers import get_ct_fullname, normalize_hus, get_bone_coords, get_all_windows, get_occ_coords
from tqdm import tqdm

__all__ = ['ManifoldCTDataset']

GENERAL_MEAN = 1.2144812352201912
GENERAL_STD = 312.14062379987047

def seed_worker_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

@dataclass
class ManifoldCTDataset(Dataset):
    """Dataset that deals with .nib CT scans"""

    # basic info
    cfg: DictConfig
    ct_numbers: List[int]
    split: str
    window_size: List[int]
    
    # data loader
    num_workers: int
    batch_size: int

    # dataset path
    dataset_path: str

    # class info
    num_classes: int
    class_encoder_map: DictConfig
    class_colors: List[str]
    class_names: List[str]

    # oversampling
    oversample_foreground_rate: float

    # legacy
    eval_only: bool = False
    stats: dict = None

    hu_mean: float = 0
    hu_std: float = 0

    def __post_init__(self):
        super().__init__()

        if self.split == 'test':
            assert self.batch_size == 1, "Batch size must be 1 for test split"

        bone_classes = OmegaConf.load("conf/classes.yaml")['bones']
        bone_classes = [k for k, v in sorted(bone_classes.items(), key=lambda item: item[1])]
        self.bones_class_indices = [self.class_names.index(c) for c in bone_classes]
        self.bones_class_indices = torch.tensor(self.bones_class_indices, dtype=torch.int64)
        self.organs_class_indices = [i for i in range(self.num_classes) if i not in self.bones_class_indices]
        self.window_size = torch.tensor(self.window_size).int()

        count = 0
        for idx in tqdm(range(len(self.ct_numbers))):
            ct_fullname = get_ct_fullname(self.ct_numbers[idx % len(self.ct_numbers)])
            raw_voxels = np.load(Path(self.dataset_path) / ct_fullname / 'sampled.npz')['voxel_features']
            self.hu_mean = (self.hu_mean * count + np.mean(raw_voxels)) / (count + 1)
            self.hu_std = (self.hu_std * count + np.std(raw_voxels)) / (count + 1)
            count += 1
        
        print("HU mean: ", self.hu_mean)
        print("HU std: ", self.hu_std)

    def collate_fn(self, batch):
        if self.split == 'train' or self.split == 'val':
            batch = [b for b in batch if b is not None and b['voxels'].coords.shape[0] > 10]
            voxels_batch = [b['voxels'] for b in batch ]
            if len(voxels_batch) == 0:
                return None

            return [{
                'voxels': Voxels.collate(voxels_batch, VoxelsFormat.FLAT),
                'ct_number': torch.cat([b['ct_number'] for b in batch], dim=0),
                'batch_size': len(voxels_batch)
            }]
        else:
            # If we are in test mode, collate is not handled here
            return batch[0]
        
    def get_trainval_window_pos(self, voxels):
        m0 = voxels.coords[..., 0].min().item()
        m1 = voxels.coords[..., 1].min().item()
        m2 = voxels.coords[..., 2].min().item()

        M0 = voxels.coords[..., 0].max().item()
        M1 = voxels.coords[..., 1].max().item()
        M2 = voxels.coords[..., 2].max().item()

        if np.random.uniform(0, 1) < 1-self.oversample_foreground_rate:
            window_pos = torch.tensor([
                np.random.randint(m0, max(M0 - self.window_size[0], m0) + 1),
                np.random.randint(m1, max(M1 - self.window_size[1], m1) + 1),
                np.random.randint(m2, max(M2 - self.window_size[2], m2) + 1)
            ], dtype=torch.int32)
        else:
            # Oversampling
            oversample_labels, oversample_coords = voxels.labels, voxels.coords

            available_classes = torch.unique(oversample_labels)
            available_classes = available_classes[available_classes > 0]
            if len(available_classes) == 0:
                window_pos = torch.tensor([m0, m1, m2], dtype=torch.int32)
            else:
                selected_class = torch.randint(0, len(available_classes), (1,)).item()
                voxel_coords_of_class = oversample_coords[oversample_labels == available_classes[selected_class]]
                if voxel_coords_of_class.shape[0] == 0:
                    window_pos = torch.tensor([m0, m1, m2], dtype=torch.int32)
                else:
                    selected_center = voxel_coords_of_class[torch.randint(0, voxel_coords_of_class.shape[0], (1,)).item()].squeeze()
                    window_pos = selected_center - self.window_size // 2
        
        return window_pos

    @torch.no_grad()
    def __getitem__(self, idx: int):
        ct_fullname = get_ct_fullname(self.ct_numbers[idx % len(self.ct_numbers)])
        raw_voxels = np.load(Path(self.dataset_path) / ct_fullname / 'sampled.npz')
        
        voxels = Voxels(format=VoxelsFormat.SINGLE, coords=torch.from_numpy(raw_voxels['voxel_coords']).int(),
                        feats=torch.from_numpy(raw_voxels['voxel_features']).float(),
                        labels=torch.from_numpy(raw_voxels['voxel_labels']).long())
        voxels.feats = normalize_hus(voxels.feats, mean=self.hu_mean, std=self.hu_std)
        ct_number = torch.tensor([self.ct_numbers[idx % len(self.ct_numbers)]], dtype=torch.int32)

        if self.split == 'train' or self.split == 'val':
            window_pos = self.get_trainval_window_pos(voxels)
            return {
                'voxels': voxels.crop_window(window_pos, self.window_size)[0],
                'ct_number': ct_number,
            }
        else:
            raise NotImplementedError("Test split not implemented yet")
            ret = get_all_windows(voxels, self.window_size)
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