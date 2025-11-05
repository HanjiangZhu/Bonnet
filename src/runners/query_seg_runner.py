import torch
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

class QuerySegRunner(BaseRunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.max_point_count = 0

        self.segmentator = hydra.utils.instantiate(cfg.seg_model,
                                                   num_classes=cfg.data.num_classes,
                                                   train_idxs=cfg.train_idxs,
                                                   _recursive_=False).to(cfg.device)
        self.query_loss_fn = OccupancySegmentationLoss(cfg=cfg, prefix='')
        self.set_losses(occ=self.query_loss_fn)

        param_count = sum(p.numel() for p in self.segmentator.parameters() if p.requires_grad) / 1e6
        print("Param count: ", param_count)
        if wandb.run is not None:
            wandb.log({"Model/ParamCount": param_count})

    def on_start_epoch(self):
        # Print LR
        if self.split == "train":
            for param_group in self.optimizer.param_groups:
                print(f"Learning rate: {param_group['lr']}")
        
        pass

    def on_end_epoch(self):
        pass

    def clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.segmentator.parameters(), 12)

    def run_model_trainval(self, batch_data):        
        assert isinstance(batch_data, dict), "batch_data must be a dictionary during train and val!"
        model = self.segmentator

        result = model(batch_data['bones'], batch_data['occ'])

        return {
            'query_logits': result['query_logits'],
            'query_labels': batch_data['occ'].labels,
            'query_mask': batch_data['occ'].mask,
            'batch_size': batch_data['batch_size'],
        }

    @torch.no_grad()
    def run_model_test(self, batch_list):
        """
        1. Processes multiple batches of the SAME sample
        2. Averages the predictions
        3. Returns the resulting logits and labels
        """

        window_size = torch.tensor(self.cfg.data.window_size, dtype=torch.float32, device=self.cfg.device)

        all_preds = []
        all_labels = []
        all_masks = []
        window_indices = []

        for batch_data in batch_list:
            preds = self.segmentator({
                'voxel_coords': batch_data['voxel_coords'],
                'voxel_features': batch_data['voxel_features'],
                'query_coords': batch_data['query_coords'],
                'query_features': batch_data['query_features'],
                'batch_size': batch_data['batch_size'],
            })

            if self.cfg.eval.gauss_kernel_inference:
                G = gaussian_kernel(batch_data['voxel_coords'], window_size, std_scale=self.cfg.eval.gauss_kernel_std_scale)
                preds *= G[:, None]

            all_preds.append(preds)
            all_labels.append(batch_data['voxel_labels'].view(-1))
            all_masks.append(batch_data['voxel_mask'].view(-1))
            window_indices.append(batch_data['window_indices'].view(-1))

        preds = torch.cat(all_preds, dim=0).float()
        labels = torch.cat(all_labels, dim=0).long()
        masks = torch.cat(all_masks, dim=0).float()
        window_indices = torch.cat(window_indices, dim=0)

        # For the testing set, ensemble the points with more than one prediction
        num_points = window_indices.max().item() + 1
        scatter_indices = window_indices.unsqueeze(1).expand(-1, preds.size(1)).long()
        final_preds = torch.zeros((num_points, preds.size(1)), dtype=torch.float32, device=preds.device)
        final_preds.scatter_reduce_(dim=0, index=scatter_indices, src=preds, reduce='sum', include_self=False)
        preds = final_preds
        # Re-normalize predictions
        # preds /= preds.sum(axis=-1, keepdims=True)
        
        final_labels = torch.zeros(num_points, dtype=torch.float32, device=labels.device)
        final_labels.scatter_reduce_(dim=0, index=window_indices.long(), src=labels.float(), reduce='mean', include_self=False)
        final_labels = final_labels.long()
        labels = final_labels

        return preds, labels, masks


    def on_start_step(self, step_counter: int, split: str, batch_list_or_dict: Dict[str, torch.Tensor]):
        if split == "train":
            self.segmentator.train()
        else:
            self.segmentator.eval()
        
        assert self.segmentator.training == (split == "train"), "Model is not in the correct mode!"
        if self.split == "train" or self.split == "val":
            result = self.run_model_trainval(batch_list_or_dict)
        else: # self.split == "test":
            raise NotImplementedError()
            # logits, labels, mask = self.run_model_test(batch_list_or_dict)
        
        # bones_loss = 0.0
        # if result['voxel_logits'] is not None:
        #     bones_loss = self.bones_loss_fn(
        #         result['voxel_logits'], result['voxel_labels'], result['batch_size']
        #     )
        
        self.max_point_count = max(self.max_point_count, result['query_logits'].shape[1])
        query_loss = self.query_loss_fn(
            result['query_logits'], result['query_labels'], result['query_mask']
        )

        return query_loss

    def log_points(self, batch_list, step_counter: int):        
        if not self.cfg.enable_vis:
            return

        MAX_BONES = 20000
        MAX_ORGANS = 60000
        
        batch_data = batch_list if isinstance(batch_list, dict) else batch_list[0]
        class_colors = np.asarray(self.cfg.data.class_colors)

        bone_points = batch_data['voxel_coords'].detach().cpu().numpy()
        bone_labels = batch_data['voxel_labels'].detach().cpu().numpy()
        bone_feats = batch_data['voxel_features'].detach().cpu().numpy()

        points = batch_data['query_coords'].detach().cpu().numpy()
        feats = batch_data['query_features'].detach().cpu().numpy()
        labels = batch_data['query_labels'].detach().cpu().numpy()

        for b in range(batch_data['batch_size']):
            bone_batch_indices = np.where(bone_points[:, 0] == b)[0]
            bone_batch_indices = np.random.choice(bone_batch_indices, size=min(MAX_BONES, bone_batch_indices.shape[0]), replace=False)
            bone_sample_points = bone_points[bone_batch_indices, 1:]
            bone_sample_labels = bone_labels[bone_batch_indices]
            bone_sample_labels = bone_sample_labels > 0 # BINARIZE BONE LABELS
            bone_sample_feats = bone_feats[bone_batch_indices]

            points_indices = np.random.choice(np.arange(points.shape[1]), size=min(MAX_ORGANS, points.shape[1]), replace=False)
            sample_points = points[b, points_indices]
            sample_labels = labels[b, points_indices]
            sample_feats = feats[b, points_indices, None]

            # Concat the points
            sample_points = np.concatenate([bone_sample_points, sample_points], axis=0)
            sample_labels = np.concatenate([bone_sample_labels, sample_labels], axis=0)
            sample_feats = np.concatenate([bone_sample_feats, sample_feats], axis=0)

            wandb.log({
                f"QueryMedia/GT_({b},{self.epoch},{step_counter})": wandbutils.create_point_cloud(sample_points, sample_labels, class_colors),
            })


    def on_end_step(self, step_counter: int, batch_list: Dict[str, torch.Tensor]):
        if self.epoch == 0 and step_counter == 0:
            self.log_points(batch_list, step_counter)