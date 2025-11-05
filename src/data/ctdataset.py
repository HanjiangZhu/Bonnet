import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
import data.data_utils as data_utils
from pathlib import Path
from p_tqdm import p_map
import logging
import numpy as np
import random
import torch
from typing import List, Dict, Tuple
from data.data_utils import dict_to_filename
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from data.torch_datatransform import torch_feature_swap, torch_flat_coords_cut_window, torch_downsample_points_if_necessary, torch_calculate_positive_shift
import logging
from .dataloader_wrapper import StepsPerEpochDataLoader, DataLoaderCUDABus
from utils.bin_packing import bin_packing
import math
from .test_patches_provider import get_all_patches
from skimage import measure

__all__ = ['MRIDataset']

# dataset characteristics according to nnunet
GENERAL_PERCENTILE_00_5 = -1003.0
GENERAL_PERCENTILE_99_5 = 1546.0
GENERAL_STD = 505.3545227050781
GENERAL_MAX = 11368.0
GENERAL_MEAN = -104.43730926513672
GENERAL_MEDIAN = 39.0
GENERAL_MIN = -9052.0

def voxel_to_contour(voxel: np.ndarray) -> np.ndarray:
    if np.all(voxel == 0):
        return None
    verts, _, _, _ = measure.marching_cubes(voxel, 0)
    return verts

def seed_worker_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

MAX_PNTS_PER_FWD_CALL = 2_000_000

BATCH_INFO = {
    # (dtype, is_contiguous)
    "voxel_coords": (torch.int32, True),
    "voxel_labels": (torch.int64, True),
    "voxel_features": (torch.float32, True),
    "voxel_mask": (torch.float32, True),
    "query_coords": (torch.int32, True),
    "query_labels": (torch.int64, True),
    "query_features": (torch.float32, True),
    "query_mask": (torch.float32, True),
    "window_indices": (torch.int64, True),
    "ct_number": (torch.int32, True),
    "indices": (torch.int32, True),
    "window_pos": (torch.int32, True),
}

# not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))


@dataclass
class CTDataset(Dataset):
    """Dataset that deals with .nib MRI scans"""

    cfg: DictConfig
    ct_numbers: List[int]
    split: str
    batch_size: int
    dataset_path: str
    cache_params: DictConfig
    cache_filename_dict: DictConfig
    caching_pool_size: int
    num_workers: int
    class_encoder_map: DictConfig
    class_colors: List[str]
    class_names: List[str]
    num_classes: int
    cache_path: str
    window_size: List[int]
    max_points_count: int
    max_points_dropout: float
    test_window_overlap: float
    test_num_point_perms: int
    test_max_points: int
    test_dropout_per_perm: int
    
    dense_voxel_path: str = None
    occupancy_lowres_factor: int = 2

    cache_name: str = field(init=False)
    stats: Dict = None
    fixate_window_to_0: bool = False

    flat_superpoints: bool = False

    oversample_foreground_rate: float = 0.33

    """
    If this value is true, we avoid loading the train/validation dataset. Instead, we only
    load cached statistics that are necessary for the normalization of the test dataset. 
    """
    eval_only: bool = False    

    hu_feature_noise_scale: float = 20.0
    window_size_range: Tuple[float, float] = (0.2, 1.0)
    max_shift: float = 0.0
    max_points_feature_swap_rate: float = 0.0
    max_points_feature_dropout_rate: float = 0.0
    feature_dropout_hu_range: Tuple[int, int] = (-100, 0)

    max_superpoints_count: int = None
    max_superpoints_feature_swap_rate: float = 0.0
    max_superpoints_feature_dropout_rate: float = 0.0

    superpoint_occlusion_rate: float = 0.0
    superpoint_occlusion_block_size: Tuple[float, float] = (0.2, 0.5)

    include_query_points: bool = False

    oversample_query_points_rate: float = 0.0
    uniform_query_points_mix: float = 0.5
    uniform_query_points_max_bg: float = 0.1

    contour_noise_std: float = 2.0
    contour_center_pull: float = 1.0

    max_query_pnt_count: int = 100_000_000

    uniform_points_only: bool = False

    def __post_init__(self):
        super().__init__()

        if self.split != 'train':
            assert self.stats is not None, "Statistics must be provided for test/val splits"
        
        if self.split == 'train' and self.split is None:
            assert len(self.ct_numbers) > 100, "Training split must have more than 100 samples"

        if self.split == 'test':
            assert self.batch_size == 1, "Test split only supports batch size 1"

        if self.cache_params.get("superpoint_radius", 0) > 0:
            assert all(x % self.cache_params.superpoint_radius == 0 for x in self.window_size)

        bone_classes = OmegaConf.load("../conf/classes.yaml")['bones']
        bone_classes = [k for k, v in sorted(bone_classes.items(), key=lambda item: item[1])]
        self.bones_class_indices = [self.class_names.index(c) for c in bone_classes]
        self.bones_class_indices = torch.tensor(self.bones_class_indices, dtype=torch.int64)
        self.organs_class_indices = [i for i in range(self.num_classes) if i not in self.bones_class_indices]

        self.cache_name = dict_to_filename(OmegaConf.to_container(self.cache_filename_dict))
        self.cache_path = Path(self.cache_path, self.cache_name)
        self.cpu_data = {}  # CPU cache of the dataset
        self.window_size = torch.tensor(self.window_size).int()
        
        logging.info(f"{self.split}: Creating cache for the uncached samples...")
        logging.info(self.cache_path)
        
        # self.hu_dist = np.zeros(self.cache_params.hu_max - self.cache_params.hu_min + 1, dtype=np.int64)
        self.hu_mean = 0.0
        self.hu_std = 1.0
        self.hu_count = 0
        
        stats_path = Path(self.cache_path, f'stats.npz')
        if stats_path.exists():
            stats = np.load(stats_path)
            self.hu_mean = stats['hu_mean']
            self.hu_std = stats['hu_std']
            self.stats = {
                'hu_mean': self.hu_mean,
                'hu_std': self.hu_std
            }

        # When in eval_only mode, skips pre-loading the training/validation dataset
        if not stats_path.exists() or not self.cfg.eval.eval_only or (self.cfg.eval.eval_only and self.split == 'test'):
            if self.caching_pool_size > 1:
                p_map(self.create_sample, range(len(self.ct_numbers)), num_cpus=self.caching_pool_size)
            else:
                for idx in tqdm(range(len(self.ct_numbers))):
                    self.create_sample(idx)
            logging.info("Validating cache and collecting statistics...")
            for idx in tqdm(range(len(self.ct_numbers))):
                self.get_sample(idx, collect_stats=True)
        
        if self.stats is None:
            self.stats = {
                'hu_mean': self.hu_mean,
                'hu_std': self.hu_std
            }
        
        if self.split == 'train' and not self.cfg.debug:
            print("[TRAIN DS] Created statistics cache at", Path(self.cache_path, f'stats.npz'))
            np.savez(Path(self.cache_path, f'stats.npz'), **self.stats)
        
        self.hu_mean = self.stats['hu_mean']
        self.hu_std = self.stats['hu_std']

        print(f"{self.split} --- dataset mean: {self.hu_mean}, std: {self.hu_std}")
        logging.info(f"Dataset cache loaded from {self.cache_path}")

    def create_sample_field(self,
                            idx: int,
                            field_name: str,
                            hu_voxels: torch.Tensor,
                            voxels_labels: torch.Tensor):
        ct = self.ct_numbers[idx]
        ct_fullname = f's{str(ct).zfill(4)}'
        cache_sample_folder = Path(self.cache_path, ct_fullname)

        if not cache_sample_folder.exists():
            os.makedirs(cache_sample_folder)
        
        if (cache_sample_folder / (field_name + '.npz')).exists() or (cache_sample_folder / (field_name + '.h5')).exists():
            return

        # Only create the versions actually requested
        if field_name == 'sparse_voxel':
            voxel_coords, voxel_labels, voxel_feats = data_utils.voxel_to_cloud(hu_voxels, voxels_labels,
                                                                               self.cache_params.hu_min,
                                                                               self.cache_params.hu_max)

            data = {
                'voxel_features': voxel_feats,
                'voxel_coords': voxel_coords,
                'voxel_labels': voxel_labels
            }
            np.savez_compressed(cache_sample_folder / "sparse_voxel.npz", **data)
        else:
            raise Exception("Invalid field name", field_name)

    def convert_dtypes_to_specs(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                dtype, contiguous = BATCH_INFO[k]
                batch[k] = batch[k].type(dtype)
                if contiguous:
                    batch[k] = batch[k].contiguous()

    def create_sample(self, idx):
        ct = self.ct_numbers[idx]
        ct_fullname = f's{str(ct).zfill(4)}'

        all_fields_exist = True
        for field in self.cache_params.batch_specs:
            if field != 'dense_voxel_features' and field != 'dense_voxel_labels' and not os.path.exists(Path(self.cache_path, ct_fullname, field + '.npz')) and not os.path.exists(Path(self.cache_path, ct_fullname, field + '.h5')):
                all_fields_exist = False
                break
        
        if all_fields_exist:
            return

        # If cache doesn't exist, process the sample
        raw_data_filepath = Path(self.dataset_path, ct_fullname)
        hu_voxels, voxels_labels = data_utils.load_ct_and_flatten_labels(raw_data_filepath, self.class_encoder_map)

        if hu_voxels is None:
            logging.info(f"Skipping (no raw data) {ct_fullname}")
            return None

        for field in self.cache_params.batch_specs:
            self.create_sample_field(idx, field, hu_voxels, voxels_labels)

    def collect_sample_stats(self, sample):
        return GENERAL_MEAN, GENERAL_STD
        # return sample['voxel_features'].mean(), sample['voxel_features'].std()

    def get_sample(self, idx: int, collect_stats=False):
        """
        Gets a sample from cache by loading each field in cache_params.batch_specs.
        """

        if idx not in self.cpu_data:
            self.cpu_data[idx] = {}
        
        result = {}
        result["ct_index"] = idx
        result["ct_number"] = self.ct_numbers[idx]

        for field_name in self.cache_params.batch_specs:
            field_data = self.get_sample_field(idx, field_name)
            
            if field_data is None:
                return None
            
            keys_to_select = { k: k for k in field_data.keys() }

            if field_name == 'sparse_voxel_superpoints':
                # For superpoints, we randomly select one of the superpoints
                if self.split == 'test':
                    grouping = f"{0,0,0}" # TODO: Use this for ensembling?
                else:
                    grouping = f"{np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2)}"
                keys_to_select = {
                    f'supervoxel_coords_{grouping}': 'supervoxel_coords',
                    f'supervoxel_features_{grouping}': 'supervoxel_features'
                }

            for key_src, key_dst in keys_to_select.items():
                assert key_dst not in result
                result[key_dst] = field_data[key_src]
        
        if collect_stats:
            # Approximate overall HU mean/std by averaging the mean/std of each sample
            hu_mean, hu_std = self.collect_sample_stats(result)
            self.hu_mean = (self.hu_mean * self.hu_count + hu_mean) / (self.hu_count + 1)
            self.hu_std = (self.hu_std * self.hu_count + hu_std) / (self.hu_count + 1)
            self.hu_count += 1

        return result
        # except Exception as e:
        #     print(e)
        #     # Delete corrupted cache directory
        #     logging.info(f">>>>>>>>> Error in: {field_name} {idx} {self.ct_numbers[idx]}")
        #     return None

    def get_sample_field(self, idx: int, field_name: str, attempt: int = 0):
        """
        Tries to load a sample field from cache. If it fails, it re-creates the sample and tries again.
        """
        if idx in self.cpu_data and field_name in self.cpu_data[idx]:
            return self.cpu_data[idx][field_name]

        ct = self.ct_numbers[idx]
        ct_fullname = f's{str(ct).zfill(4)}'

        if field_name == 'dense_voxel_features':
            self.cpu_data[idx][field_name] = {
                'dense_voxel_features': torch.from_numpy(np.load(Path(self.dense_voxel_path) / ct_fullname / 'ct.npz')['arr_0']).float()
            }
            return self.cpu_data[idx][field_name]
        elif field_name == 'dense_voxel_labels':
            self.cpu_data[idx][field_name] = {
                'dense_voxel_labels': torch.from_numpy(np.load(Path(self.dense_voxel_path) / ct_fullname / 'labels.npz')['arr_0']).long()
            }
            return self.cpu_data[idx][field_name]

        if attempt > 1:
            return None

        cache_sample_folder = Path(self.cache_path) / ct_fullname

        try:
            npy_field_data = np.load(cache_sample_folder / (field_name + '.npz'), allow_pickle=True)
            torch_field_data = {}
            for k in npy_field_data.files:
                torch_field_data[k] = torch.from_numpy(npy_field_data[k])
            return torch_field_data
        except Exception as e:
            logging.error(e)
            logging.error(f"[!!!] Corrupted cache file {cache_sample_folder}. Error in field {field_name}. Re-creating sample...")
            # Delete corrupted cache file
            os.remove(cache_sample_folder / (field_name + '.npz'))
            self.create_sample(idx)
            return self.get_sample_field(idx, field_name, attempt=attempt + 1)

    def collate_samples(self, samples_list):
        samples_list = [ x for x in samples_list if x is not None and x['voxel_coords'].shape[0] > 100 ]
        point_counts = [x['voxel_coords'].shape[0] for x in samples_list]
        
        if len(samples_list) == 0:
            logging.warn("Found a batch with invalid point counts: ", point_counts)
            return None
    
        max_pnts_per_fwd_call = MAX_PNTS_PER_FWD_CALL
        
        if self.split == 'test':
            max_pnts_per_fwd_call = max_pnts_per_fwd_call // int(self.cfg.seg_model.width)
        
        if self.split == 'train':
            bins = [{"indices": list(range(len(samples_list)))}]
        else:
            bins = bin_packing(point_counts, max_pnts_per_fwd_call)
       
        batches = []
        for bin in bins:
            batch = self._internal_collate_samples([samples_list[i] for i in bin["indices"]])
            self.convert_dtypes_to_specs(batch)
            batches.append(batch)
        
        return batches

    def _internal_collate_samples(self, samples_list):
        """
        Collates a list of samples (dicts) into a batch.
        """

        voxel_coords_list = [x['voxel_coords'] for x in samples_list]

        voxel_batch_idxs = torch.arange(len(voxel_coords_list), dtype=torch.int32)
        voxel_batch_idxs = torch.repeat_interleave(voxel_batch_idxs, torch.tensor([len(x) for x in voxel_coords_list], dtype=voxel_batch_idxs.dtype))
        voxel_coords = torch.concatenate([x for x in voxel_coords_list], axis=0)
        voxel_coords = torch.concatenate([voxel_batch_idxs[:, None], voxel_coords], axis=1)

        result = {
            k: torch.concatenate([x[k] for x in samples_list], axis=0) for k in samples_list[0].keys() if not k.endswith('coords')
        }
        result['voxel_coords'] = voxel_coords.int()
        result['voxel_features'] = result['voxel_features'].float()
        result['voxel_labels'] = result['voxel_labels'].long()
        result['batch_size'] = len(samples_list)
        return result

    def collate_fn(self, batch):
        """
        Collate function for the dataloader. It takes a list of samples and returns a batch.
        In data: list of [ voxel_coords, voxel_features, voxel_labels ]
              Each item in the list has shape:
                voxel_coords has shape (N, 3)
                voxel_features has shape (N, C)
        Out data: dict of [ voxel_coords, voxel_features, voxel_labels ]
              voxel_coords has shape (N, 4 = batch_idx + x + y + z)
              voxel_features has shape (N, C)
        """
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            logging.warn("Every sample in this batch is None")
            return None
    
        if self.split == "test": # NOTE: samples are pre-collated for testing
            assert len(batch) == 1
            return batch[0]

        collated_batch = self.collate_samples(batch)
        return collated_batch

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

        # ds_loader = DataLoaderCUDABus(ds_loader, enable_prefetch=enable_data_prefetch)
        return ds_loader
    
    def get_single_patch(self, voxel_coords, voxel_features, voxel_labels):
        if voxel_coords.shape[0] < 100:
            return None

        voxel_coords = voxel_coords.int()
        voxel_labels = voxel_labels.long()

        m0 = voxel_coords[..., 0].min().item()
        m1 = voxel_coords[..., 1].min().item()
        m2 = voxel_coords[..., 2].min().item()

        if self.fixate_window_to_0:
            window_pos = torch.tensor([m0, m1, m2], dtype=torch.int32)
        else:
            if np.random.uniform(0, 1) < 1-self.oversample_foreground_rate:
                window_pos = torch.tensor([
                    np.random.randint(m0, max(voxel_coords[..., 0].max() - self.window_size[0], m0) + 1),
                    np.random.randint(m1, max(voxel_coords[..., 1].max() - self.window_size[1], m1) + 1),
                    np.random.randint(m2, max(voxel_coords[..., 2].max() - self.window_size[2], m2) + 1)
                ], dtype=torch.int32)
            else:
                available_classes = torch.unique(voxel_labels)
                available_classes = available_classes[(available_classes != 0) & torch.isin(available_classes, self.bones_class_indices)]
                if len(available_classes) == 0:
                    window_pos = torch.tensor([m0, m1, m2], dtype=torch.int32)
                else:
                    selected_class = torch.randint(0, len(available_classes), (1,)).item()
                    voxel_coords_of_class = voxel_coords[voxel_labels == available_classes[selected_class]]
                    if voxel_coords_of_class.shape[0] == 0:
                        window_pos = torch.tensor([m0, m1, m2], dtype=torch.int32)
                    else:
                        selected_center = voxel_coords_of_class[torch.randint(0, voxel_coords_of_class.shape[0], (1,)).item()].squeeze()
                        window_pos = selected_center - self.window_size // 2

        voxel_coords, voxel_features, voxel_labels = torch_flat_coords_cut_window(voxel_coords,
                                                                                  voxel_features,
                                                                                  voxel_labels, self.window_size, window_pos)

        if voxel_coords.shape[0] < 30:
            return None

        result = {
            'voxel_coords': voxel_coords,
            'voxel_features': voxel_features,
            'voxel_labels': voxel_labels,
            'window_pos': window_pos
        }
        return result

    def get_num_patches(self, coords):
        num_width_patches = coords[:, 0].max().item() / self.window_size[0]
        num_height_patches = coords[:, 1].max().item() / self.window_size[1]
        num_depth_patches = coords[:, 2].max().item() / self.window_size[2]

        if self.test_window_overlap > 0.0:
            num_width_patches *= (1 / (1 - self.test_window_overlap))
            num_height_patches *= (1 / (1 - self.test_window_overlap))
            num_depth_patches *= (1 / (1 - self.test_window_overlap))
        
        num_width_patches = int(math.ceil(num_width_patches))
        num_height_patches = int(math.ceil(num_height_patches))
        num_depth_patches = int(math.ceil(num_depth_patches))

        return num_width_patches, num_height_patches, num_depth_patches

    def get_all_patches(self, voxel_coords, voxel_features, voxel_labels):
        assert self.batch_size == 1

        all_samples = get_all_patches(voxel_coords, voxel_features, voxel_labels, self.window_size, self.test_window_overlap)

        if len(all_samples) == 0:
            return None
        else:
            all_samples = self.collate_samples(all_samples)
            return all_samples

    @torch.no_grad()
    def downsample_patch(self, voxel_dict: dict):
        if voxel_dict is None:
            return None

        max_points_count = min(self.max_points_count, voxel_dict['voxel_coords'].shape[0])
        if self.max_points_dropout > 0:
            max_points_count *= (1 - np.random.uniform(0, self.max_points_dropout))
        max_points_count = int(max_points_count)

        torch_downsample_points_if_necessary(max_points_count, voxel_dict)

    @torch.no_grad()
    def apply_augmentations(self, voxel_dict):
        min_coords = voxel_dict['voxel_coords'].min(axis=0, keepdims=True).values
        voxel_dict['voxel_coords'] = voxel_dict['voxel_coords'] - min_coords
        shift = torch_calculate_positive_shift(voxel_dict['voxel_coords'], self.window_size)
        voxel_dict['voxel_coords'] += shift

        if self.max_points_feature_swap_rate > 0:
            voxel_dict['voxel_features'] = torch_feature_swap(voxel_dict['voxel_features'], self.max_points_feature_swap_rate)

    @torch.no_grad()
    def __getitem__(self, idx: int):
        idx = idx % len(self.ct_numbers)
        input_batch = self.get_sample(idx)
        initial_coords_count = input_batch['voxel_coords'].shape[0]

        if input_batch is None or len(input_batch) == 0 or 'is_invalid' in input_batch:
            return None
        
        voxel_coords, voxel_features, voxel_labels = input_batch['voxel_coords'], input_batch['voxel_features'], input_batch['voxel_labels']
        voxel_features = voxel_features[..., None].float()
        voxel_features = (voxel_features - self.hu_mean) / self.hu_std

        voxel_result = None
        if self.split == 'train' or self.split == 'val':
            voxel_result = self.get_single_patch(voxel_coords, voxel_features, voxel_labels)
            if voxel_result is not None:
                self.downsample_patch(voxel_result)
                self.apply_augmentations(voxel_result)
        else:
            voxel_result = self.get_all_patches(voxel_coords, voxel_features, voxel_labels)
        
        assert initial_coords_count == input_batch['voxel_coords'].shape[0], "Input batch was modified"
        return voxel_result

    def __len__(self):
        L = len(self.ct_numbers)
        return L
