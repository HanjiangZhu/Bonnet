import torch
from typing import Dict
from omegaconf import DictConfig
import metrics.wandbutils as wandbutils
import numpy as np
import wandb
import hydra
from runners.base_runner import BaseRunner
import torch.nn.functional as F
from models.models_utils import gaussian_kernel

class CNN_SegRunner(BaseRunner):
    def __init__(self, cfg: DictConfig, loss_cfg: DictConfig):
        super().__init__(cfg)
        self.segmentator = hydra.utils.instantiate(cfg.seg_model, num_classes=cfg.data.num_classes, _recursive_=False).to(cfg.device)
        self.loss_fn = hydra.utils.instantiate(loss_cfg, cfg=cfg, _recursive_=False)
        self.set_losses(seg=self.loss_fn)

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
        window_indices = []

        for batch_data in batch_list:
            preds = self.segmentator(batch_data)

            if self.cfg.eval.gauss_kernel_inference:
                G = gaussian_kernel(batch_data['voxel_coords'], window_size, std_scale=self.cfg.eval.gauss_kernel_std_scale)
                preds *= G[:, None]

            all_preds.append(preds)
            all_labels.append(batch_data['voxel_labels'].view(-1))
            window_indices.append(batch_data['window_indices'].view(-1))
                
        preds = torch.cat(all_preds, dim=0).float()
        labels = torch.cat(all_labels, dim=0).long()
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

        return preds, labels, 1
    
    def run_model_trainval(self, batch_data):
        assert isinstance(batch_data, dict), "batch_data must be a dictionary during train and val!"
        model = self.segmentator
        logits = model(batch_data)
        labels = batch_data['voxel_labels']
        batch_size = batch_data['batch_size']
        return logits, labels, batch_size

    def on_start_step(self, step_counter: int, split: str, batch_list_or_dict: Dict[str, torch.Tensor]):
        if split == "train":
            self.segmentator.train()
        else:
            self.segmentator.eval()
        
        assert self.segmentator.training == (split == "train"), "Model is not in the correct mode!"
        
        if self.split == "train" or self.split == "val":
            logits, labels, batch_size = self.run_model_trainval(batch_list_or_dict)
        else: # self.split == "test":
            logits, labels, batch_size = self.run_model_test(batch_list_or_dict)
        
        loss = self.loss_fn(logits, labels,
                            batch_size=batch_size)

        return loss

    def on_end_step(self, step_counter: int, batch_list: Dict[str, torch.Tensor]):
        if step_counter == 0 and self.epoch == 0 and self.cfg.use_wandb and self.split == "train":
            batch_data = batch_list[0]
            points = batch_data['voxel_coords'].detach().cpu().numpy()
            feats = batch_data['voxel_features'].detach().cpu().numpy()
            labels = batch_data['voxel_labels'].detach().cpu().numpy()
            class_colors = np.asarray(self.cfg.data.class_colors)

            for b in range(min(3, batch_data['batch_size'])):
                batch_0_indices = np.where(points[:, 0] == b)[0]
                batch_0_indices = np.random.choice(batch_0_indices, size=50000, replace=False)
                sample_points = points[batch_0_indices][:, 1:]
                sample_labels = labels[batch_0_indices]

                # Concat with feats
                sample_feats = feats[batch_0_indices]
                sample_points_with_hu = np.concatenate([sample_points, sample_feats], axis=1)

                wandb.log({
                    f"GT_({b},{self.epoch},{step_counter})": wandbutils.create_point_cloud(sample_points, sample_labels, class_colors),
                    f"HU_({b},{self.epoch},{step_counter})": wandbutils.create_point_cloud(sample_points_with_hu, None, class_colors),
                })