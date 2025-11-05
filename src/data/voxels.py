import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from .data_utils import collate_coords_flat
import numpy as np
import metrics.wandbutils as wandbutils


# In flat mode, coordinates have the form [Batch Index, X, Y, Z]

class VoxelsFormat(Enum):
    BATCHED = 1 # (B, N, C) -> PointNet, PointNet++, PVCNN++
    FLAT = 2    # (M, C) -> A1, A2, A3, ..., AN, B1, B2, B3, ..., BM -> Sparse UNet (Submanifold convolution)
    SINGLE = 3  # (N, C) -> Single sample


@dataclass
class Voxels:
    @staticmethod
    def from_dense(coords: torch.Tensor, vox_feats: Optional[torch.Tensor] = None, vox_labels: Optional[torch.Tensor] = None):
        return Voxels(format=VoxelsFormat.SINGLE,
                      coords=coords,
                      feats=vox_feats[coords[:, 0], coords[:, 1], coords[:, 2]] if vox_feats is not None else None,
                      labels=vox_labels[coords[:, 0], coords[:, 1], coords[:, 2]] if vox_labels is not None else None)

    @staticmethod
    def collate(voxels_list, format: VoxelsFormat):
        for x in voxels_list:
            assert x.format == VoxelsFormat.SINGLE, "Only single voxels can be collated"
        for i in range(len(voxels_list)): voxels_list[i].create_mask()

        if format == VoxelsFormat.BATCHED:
            max_num_points = max([x.coords.shape[0] for x in voxels_list])
            same_size_list = [x.ensure_size(max_num_points) for x in voxels_list]
            
            coords = torch.stack([x.coords for x in same_size_list], dim=0) if same_size_list[0].coords is not None else None
            labels = torch.stack([x.labels for x in same_size_list], dim=0) if same_size_list[0].labels is not None else None
            feats = torch.stack([x.feats for x in same_size_list], dim=0) if same_size_list[0].feats is not None else None
            mask = torch.stack([x.mask for x in same_size_list], dim=0) if same_size_list[0].mask is not None else None
        else: # format == VoxelsFormat.FLAT
            coords = collate_coords_flat([x.coords for x in voxels_list]) if voxels_list[0].coords is not None else None
            labels = torch.cat([x.labels for x in voxels_list], dim=0) if voxels_list[0].labels is not None else None
            feats = torch.cat([x.feats for x in voxels_list], dim=0) if voxels_list[0].feats is not None else None
            mask = torch.cat([x.mask for x in voxels_list], dim=0) if voxels_list[0].mask is not None else None

            if len(feats.shape) == 1:
                feats = feats.unsqueeze(1)

        return Voxels(format=format, coords=coords, labels=labels, feats=feats, mask=mask)


    format: VoxelsFormat = VoxelsFormat.BATCHED
    coords: Optional[torch.Tensor] = None # 4 for flat, 3 for single
    labels: Optional[torch.Tensor] = None # 1 value for each point
    feats: Optional[torch.Tensor] = None # C channels for each point (HU value)
    mask: Optional[torch.Tensor] = None # 1 value for each point

    def clone(self):
        return Voxels(format=self.format,
                      coords=self.coords.clone() if self.coords is not None else None,
                      labels=self.labels.clone() if self.labels is not None else None,
                      feats=self.feats.clone() if self.feats is not None else None,
                      mask=self.mask.clone() if self.mask is not None else None)

    def to_wandb(self, class_colors, b: int = 0):
        if self.format == VoxelsFormat.FLAT:
            batch_0_indices = np.where(points[:, 0] == b)[0]
            batch_0_indices = np.random.choice(batch_0_indices, size=50000, replace=False)
            sample_points = points[batch_0_indices][:, 1:]
            sample_labels = labels[batch_0_indices]

            # Concat with feats
            sample_feats = feats[batch_0_indices]
            sample_points_with_hu = np.concatenate([sample_points, sample_feats], axis=1)
        elif self.format == VoxelsFormat.SINGLE:
            sample_points = self.coords.cpu().numpy()
            sample_labels = self.labels.cpu().numpy()
            sample_feats = self.feats.cpu().numpy()
            sample_points_with_hu = np.concatenate([sample_points, sample_feats], axis=1)
        else:
            sample_points = self.coords[b].cpu().numpy()
            sample_labels = self.labels[b].cpu().numpy()
            sample_feats = self.feats[b].cpu().numpy()[:, None]
            sample_points_with_hu = np.concatenate([sample_points, sample_feats], axis=1)
        
        gt = wandbutils.create_point_cloud(sample_points, sample_labels, class_colors)
        hu = wandbutils.create_point_cloud(sample_points_with_hu, None, class_colors)
        return gt, hu    


    def to_dict(self, name: str):
        result = {}
        if self.coords is not None:
            result[name + '_coords'] = self.coords
        if self.labels is not None:
            result[name + '_labels'] = self.labels
        if self.feats is not None:
            result[name + '_features'] = self.feats
        if self.mask is not None:
            result[name + '_mask'] = self.mask
        return result

    def to(self, device: str, **kwargs):
        self.coords = self.coords.to(device, **kwargs) if self.coords is not None else None
        if self.labels is not None:
            self.labels = self.labels.to(device, **kwargs)
        if self.feats is not None:
            self.feats = self.feats.to(device, **kwargs)
        if self.mask is not None:
            self.mask = self.mask.to(device, **kwargs)
        return self
    
    def to_flat_format(self):
        assert self.format == VoxelsFormat.BATCHED, "Only batched voxels can be converted to flat format. Got format: " + str(self.format)

        batch_indices = torch.arange(self.coords.shape[0], device=self.coords.device)
        new_coords = self.coords.view(-1, 3) if self.coords is not None else None
        batch_indices = batch_indices[:, None].expand(-1, self.coords.shape[1]).reshape(-1)[:, None]
        new_coords = torch.cat([batch_indices, new_coords], dim=1)
        new_labels = self.labels.view(-1) if self.labels is not None else None
        new_feats = self.feats.view(-1, self.feats.shape[2]) if self.feats is not None else None
        new_mask = self.mask.view(-1) if self.mask is not None else None
        new_format = VoxelsFormat.FLAT
        return Voxels(format=new_format, coords=new_coords, labels=new_labels, feats=new_feats, mask=new_mask)

    def crop_window(self, window_pos: torch.Tensor, window_size: torch.Tensor) -> Tuple['Voxels', torch.Tensor]:
        assert self.format == VoxelsFormat.SINGLE, "Only single voxels can be cropped"
        coords_in_window_mask = (self.coords[:, 0] >= window_pos[0]) & (self.coords[:, 0] < window_pos[0] + window_size[0]) & \
                                (self.coords[:, 1] >= window_pos[1]) & (self.coords[:, 1] < window_pos[1] + window_size[1]) & \
                                (self.coords[:, 2] >= window_pos[2]) & (self.coords[:, 2] < window_pos[2] + window_size[2])
        window_idxs = torch.arange(self.coords.shape[0])[coords_in_window_mask]
        return Voxels(format=VoxelsFormat.SINGLE,
                      coords=self.coords[coords_in_window_mask] - window_pos,
                      labels=self.labels[coords_in_window_mask] if self.labels is not None else None,
                      feats=self.feats[coords_in_window_mask] if self.feats is not None else None,
                      mask=self.mask[coords_in_window_mask] if self.mask is not None else None), window_idxs

    def create_mask(self):
        assert self.format == VoxelsFormat.SINGLE, "Only single voxels can have masks"
        if self.mask is None:
            self.mask = torch.ones((self.coords.shape[0],), dtype=torch.bool)
    
    def ensure_size(self, size: int):
        assert self.format == VoxelsFormat.SINGLE, "Only single voxels can be padded"
        num_points = self.coords.shape[0]
        repeat_coords, repeat_labels, repeat_feats, repeat_mask = None, None, None, None

        if num_points < size:
            if self.coords is not None:
                repeat_coords = torch.zeros((size, *self.coords.shape[1:]), dtype=self.coords.dtype)
                repeat_coords[:self.coords.shape[0],] = self.coords
                if self.coords.shape[0] > 0: repeat_coords[self.coords.shape[0]:,] = self.coords[0,]
            
            if self.labels is not None:
                repeat_labels = torch.zeros((size,), dtype=self.labels.dtype)
                repeat_labels[:self.labels.shape[0],] = self.labels
                if self.labels.shape[0] > 0: repeat_labels[self.labels.shape[0]:,] = self.labels[0,]
            
            if self.feats is not None:
                repeat_feats = torch.zeros((size, *self.feats.shape[1:]), dtype=self.feats.dtype)
                repeat_feats[:self.feats.shape[0],] = self.feats
                if self.feats.shape[0] > 0: repeat_feats[self.feats.shape[0]:,] = self.feats[0,]
            
            if self.mask is not None:
                repeat_mask = torch.zeros((size,), dtype=self.mask.dtype)
                repeat_mask[:self.mask.shape[0],] = self.mask
                if self.mask.shape[0] > 0: repeat_mask[self.mask.shape[0]:,] = False
        else:
            random_idxs = torch.randperm(self.coords.shape[0])[:size]
            repeat_coords = self.coords[random_idxs] if self.coords is not None else None
            repeat_labels = self.labels[random_idxs] if self.labels is not None else None
            repeat_feats = self.feats[random_idxs] if self.feats is not None else None
            repeat_mask = self.mask[random_idxs] if self.mask is not None else None

        return Voxels(format=VoxelsFormat.SINGLE, coords=repeat_coords, labels=repeat_labels, feats=repeat_feats, mask=repeat_mask)
