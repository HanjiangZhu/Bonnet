import torch
import nibabel as nib
import numpy as np
from metrics import wandbutils
from typing import Dict
from omegaconf import DictConfig
import wandb
import hydra
from runners.base_runner import BaseRunner
from models.models_utils import gaussian_kernel
from losses.occ_seg_loss import OccupancySegmentationLoss
from losses.seg_loss import SegmentationLoss
from typing import List, Tuple
from pathlib import Path
from torch_helpers.torch_voxels import torch_batched_voxelize
from data.voxels import Voxels, VoxelsFormat
from tqdm import tqdm
from data.simple_cthelpers import GENERAL_MEAN, GENERAL_STD

def torch_simple_voxelize(VoxSize, coords, feats):
    print("Performing voxelize on", VoxSize, coords.shape, feats.shape)
    return torch_batched_voxelize(feats[None,].transpose(2,1),
                                  coords[None,].float().transpose(2,1) / VoxSize,
                                  VoxSize, "mean")[0,]

def save_nib(data: np.ndarray, path: str):
    nib.Nifti1Image(data, affine=None).to_filename(path)

class ExporterSegRunner(BaseRunner):
    def __init__(self, cfg: DictConfig, num_exports: int):
        super().__init__(cfg)

        self.num_exports = num_exports
        self.exports_counter = 0
        self.segmentator = hydra.utils.instantiate(cfg.seg_model,
                                                   num_classes=cfg.data.num_classes,
                                                   superpoint_radius=1,
                                                   _recursive_=False).to(cfg.device)
        self.query_loss_fn = OccupancySegmentationLoss(cfg=cfg, prefix='')
        self.bones_loss_fn = SegmentationLoss(cfg=cfg, label_smoothing=0)
        self.set_losses(seg=self.bones_loss_fn, occ=self.query_loss_fn)

        param_count = sum(p.numel() for p in self.segmentator.parameters() if p.requires_grad) / 1e6
        print("Param count: ", param_count)
        if wandb.run is not None:
            wandb.log({"Model/ParamCount": param_count})

    def on_start_epoch(self):
        pass

    def on_end_epoch(self):
        pass

    def clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.segmentator.parameters(), 12)

    def on_start_step(self, step_counter: int, split: str, batch_list_or_dict: Dict[str, torch.Tensor]):
        self.segmentator.eval()
        
        if self.split == "train" or self.split == "val":
            raise ValueError("This runner is only for test split")
        # else self.split == "test":
        
        hus, gt_img, pred_img = self.run_model_test(batch_list_or_dict)
        
        ct_number = str(batch_list_or_dict[0]['ct_number'])
        ct_number = f's{str(ct_number).zfill(4)}_step' + str(step_counter)
        
        output_dir = Path('/netscratch/martelleto/cv3d/exports/') / ct_number
        output_dir.mkdir(parents=True, exist_ok=True)
        save_nib(hus.int().numpy(), output_dir / 'ct.nii.gz')
        save_nib(gt_img.int().numpy(), output_dir / f'gt.nii.gz')
        save_nib(pred_img.int().numpy(), output_dir / f'pred.nii.gz')
        print(f"Exported {ct_number} to {output_dir}")

        self.exports_counter += 1
        if self.exports_counter >= self.num_exports:
            print(self.query_loss_fn.metrics)
            print(f"Finished exporting {self.num_exports} samples")
            exit(0)
        
        return torch.tensor(0.0, device=self.cfg.device)

    def scatter(self, num_points, preds, scatter_indices):
        final_preds_shape = list(preds.shape)
        final_preds_shape[0] = num_points
        final_preds = torch.zeros(final_preds_shape, dtype=preds.dtype, device=preds.device)

        if scatter_indices.ndim < preds.ndim:
            scatter_indices = scatter_indices.unsqueeze(1)
        
        scatter_indices = scatter_indices.expand_as(preds)
        final_preds.scatter_reduce_(dim=0, index=scatter_indices, src=preds, reduce='mean', include_self=False)
        return final_preds

    @torch.no_grad()
    def run_model_test(self, batch_list: List[Tuple[Voxels, Voxels]]):
        """
        1. Processes multiple batches of the SAME sample
        2. Averages the predictions
        3. Returns the resulting logits and labels
        """

        window_size = torch.tensor(self.cfg.data.window_size, dtype=torch.float32, device=self.cfg.device)
        all_bones_logits = []
        all_occ_logits = []
        all_bones_indices = []
        all_occ_indices = []
        all_bones = []
        all_occ = []

        for batch_data in tqdm(batch_list):
            bones = batch_data['bones']
            occ = batch_data['occ']
            bones = Voxels.collate([bones], VoxelsFormat.FLAT).to(self.cfg.device)
            occ = Voxels.collate([occ], VoxelsFormat.BATCHED).to(self.cfg.device)
            result = self.segmentator(bones, occ)

            if self.cfg.eval.gauss_kernel_inference:
                G = gaussian_kernel(bones.coords, window_size, std_scale=self.cfg.eval.gauss_kernel_std_scale)
                result['voxel_logits'] *= G[:, None]
            
            bones = bones.to('cpu')
            occ = occ.to('cpu')

            all_bones_logits.append(result['voxel_logits'].detach().cpu())
            all_occ_logits.append(result['query_logits'].detach().cpu().squeeze(0))
            all_bones_indices.append(batch_data['bones_window_indices'].cpu())
            all_occ_indices.append(batch_data['occ_window_indices'].cpu())

            bones.coords = bones.coords[..., 1:]
            bones.format = VoxelsFormat.SINGLE
            bones.coords += batch_data['window_pos'].cpu()

            occ.coords = occ.coords[0,...]
            occ.feats = occ.feats[0,...]
            occ.labels = occ.labels[0,...]
            occ.mask = occ.mask[0,...]
            occ.format = VoxelsFormat.SINGLE
            assert torch.all(occ.coords < 128)
            occ.coords += batch_data['window_pos'].cpu()

            all_bones.append(bones)
            all_occ.append(occ)

        down_res = self.cfg.data.occ_lowres_factor

        all_bones_indices = torch.cat(all_bones_indices, dim=0)
        all_occ_indices = torch.cat(all_occ_indices, dim=0)

        all_bones_logits = torch.cat(all_bones_logits, dim=0)
        all_occ_logits = torch.cat(all_occ_logits, dim=0)
        
        all_bones = Voxels.collate(all_bones, VoxelsFormat.FLAT)
        all_bones.coords = all_bones.coords[..., 1:] // down_res

        all_occ = Voxels.collate(all_occ, VoxelsFormat.FLAT)
        all_occ.coords = all_occ.coords[..., 1:] // down_res

        num_points = all_bones_indices.max().item() + 1
        all_bones.coords = self.scatter(num_points, all_bones.coords, all_bones_indices).long()
        all_bones.feats = self.scatter(num_points, all_bones.feats, all_bones_indices)
        all_bones.labels = self.scatter(num_points, all_bones.labels, all_bones_indices).long()
        all_bones_logits = self.scatter(num_points, all_bones_logits, all_bones_indices).float()
        
        all_occ.coords = self.scatter(num_points, all_occ.coords, all_occ_indices).long()
        all_occ.feats = self.scatter(num_points,  all_occ.feats, all_occ_indices)
        all_occ.labels = self.scatter(num_points, all_occ.labels, all_occ_indices).long()
        all_occ_logits = self.scatter(num_points, all_occ_logits, all_occ_indices).float()

        VoxSize = max(all_bones.coords.max().item(), all_occ.coords.max().item()) + 1

        hus_voxel = torch_simple_voxelize(VoxSize, all_occ.coords, all_occ.feats.float().unsqueeze(1))[0,]
        preds_voxel = torch_simple_voxelize(VoxSize, all_occ.coords, all_occ_logits.float()).long().argmax(0)
        preds_voxel[all_bones.coords[:, 0], all_bones.coords[:, 1], all_bones.coords[:, 2]] = all_bones_logits.long().argmax(-1)
        gt_voxel = torch_simple_voxelize(VoxSize, all_occ.coords, all_occ.labels.float().unsqueeze(1)).long()[0,]

        # TODO: handle out-of-bounds points + is this working?

        return hus_voxel, gt_voxel, preds_voxel

    # def log_points(self, batch_list, step_counter: int):        
    #     if not self.cfg.enable_vis:
    #         return

    #     MAX_BONES = 20000
    #     MAX_ORGANS = 60000
        
    #     batch_data = batch_list if isinstance(batch_list, dict) else batch_list[0]
    #     class_colors = np.asarray(self.cfg.data.class_colors)

    #     bone_points = batch_data['voxel_coords'].detach().cpu().numpy()
    #     bone_labels = batch_data['voxel_labels'].detach().cpu().numpy()
    #     bone_feats = batch_data['voxel_features'].detach().cpu().numpy()

    #     points = batch_data['query_coords'].detach().cpu().numpy()
    #     feats = batch_data['query_features'].detach().cpu().numpy()
    #     labels = batch_data['query_labels'].detach().cpu().numpy()

    #     for b in range(batch_data['batch_size']):
    #         bone_batch_indices = np.where(bone_points[:, 0] == b)[0]
    #         bone_batch_indices = np.random.choice(bone_batch_indices, size=min(MAX_BONES, bone_batch_indices.shape[0]), replace=False)
    #         bone_sample_points = bone_points[bone_batch_indices, 1:]
    #         bone_sample_labels = bone_labels[bone_batch_indices]
    #         bone_sample_labels = bone_sample_labels > 0 # BINARIZE BONE LABELS
    #         bone_sample_feats = bone_feats[bone_batch_indices]

    #         points_indices = np.random.choice(np.arange(points.shape[1]), size=min(MAX_ORGANS, points.shape[1]), replace=False)
    #         sample_points = points[b, points_indices]
    #         sample_labels = labels[b, points_indices]
    #         sample_feats = feats[b, points_indices, None]

    #         # Concat the points
    #         sample_points = np.concatenate([bone_sample_points, sample_points], axis=0)
    #         sample_labels = np.concatenate([bone_sample_labels, sample_labels], axis=0)
    #         sample_feats = np.concatenate([bone_sample_feats, sample_feats], axis=0)

    #         wandb.log({
    #             f"QueryMedia/GT_({b},{self.epoch},{step_counter})": wandbutils.create_point_cloud(sample_points, sample_labels, class_colors),
    #         })

    def on_end_step(self, step_counter: int, batch_list: Dict[str, torch.Tensor]):
        pass
        # if self.epoch == 0 and step_counter == 0:
        #     self.log_points(batch_list, step_counter)