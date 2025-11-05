from omegaconf import DictConfig
import torch.nn as nn
import torch
from tqdm import tqdm
from typing import Dict
import numpy as np
import wandb
import logging
from data.voxels import Voxels
from data.dataloader_wrapper import DataLoaderCUDABus

def print_metrics_to_terminal(metrics):
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            metrics[k] = float(v.item())

    items = list(metrics.items())
    items_to_print = [ (k, v) for k, v in items if isinstance(v, float) ]
    for i in range(len(items_to_print)):
        key, value = items_to_print[i]
        print(f"{key}: {value:.4f}", end=", " if i < len(items_to_print) - 1 else "\n")

# TODO
# self.initial_lr = 1e-2
# self.weight_decay = 3e-5
# self.oversample_foreground_percent = 0.33
# self.num_epochs = 1000
# self.num_iterations_per_epoch = 250

class BaseRunner(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.losses = {}
        self.ema_losses = {}

    def set_losses(self, **losses: Dict[str, torch.Tensor]):
        for k, v in losses.items():
            self.losses[k] = v

    def clip_grad(self):
        raise NotImplementedError

    def run_epoch(self, split: str, data_loader, optimizer, grad_scaler,
                  lr_scheduler, epoch: int, log_to_wandb: bool = True,
                  print_to_console: bool = True):
        assert len(self.losses) > 0, "No losses set for the runner!"
        assert split in ["train", "val", "test"], "Invalid split!"

        self.split = split
        self.epoch = epoch
        self.dataloader = data_loader
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler

        prev_state = torch.is_grad_enabled()
        torch.set_grad_enabled((split == "train") and (optimizer is not None))

        if split == "train" and optimizer is not None:
            self.train()
        else:
            self.eval()

        for v in self.losses.values():
            v.on_start_epoch()
        
        self.on_start_epoch()
        logging.info(f"Starting epoch in split {split} with optimizer = {optimizer is not None}")

        if hasattr(self.dataloader, "on_start_epoch"):
            self.dataloader.on_start_epoch()

        for (step_counter, batch_list) in tqdm(enumerate(self.dataloader), smoothing=0.9, total=len(self.dataloader)):
            if batch_list is None:
                continue
            
            if self.split != 'test':
                batch_list = [{k: v.to(self.cfg.device, non_blocking=True) if torch.is_tensor(v) or isinstance(v, Voxels) else v for k, v in batch_data.items()} for batch_data in batch_list]


            if split == 'test': # Split is test, pass entire batch list
                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    with torch.no_grad():
                        total_loss = self.on_start_step(step_counter, split, batch_list)
                        
                        if split not in self.ema_losses:
                            self.ema_losses[split] = total_loss.detach()

                        self.ema_losses[split] = 0.95 * self.ema_losses[split] + 0.05 * total_loss.detach()
            else: # Split is train or val, pass each batch data separately
                for batch_data in batch_list:
                    if split == "train" and optimizer is not None:
                        optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                        total_loss = self.on_start_step(step_counter, split, batch_data)
                        
                        if total_loss is not None:
                            if split not in self.ema_losses:
                                self.ema_losses[split] = total_loss.detach()

                            self.ema_losses[split] = 0.95 * self.ema_losses[split] + 0.05 * total_loss.detach()

                            if split == "train" and optimizer is not None:
                                grad_scaler.scale(total_loss).backward()
                                grad_scaler.unscale_(optimizer)
                                self.clip_grad()
                                grad_scaler.step(optimizer)
                                grad_scaler.update()

                                if lr_scheduler is not None:
                                    lr_scheduler.step()
            
            self.on_end_step(step_counter, batch_list)

        logging.info(f"End of epoch {epoch} from split {split}.")

        for v in self.losses.values():
            v.on_end_epoch()
        
        self.on_end_epoch()
        if self.split == "train" and optimizer is not None and hasattr(lr_scheduler, "on_end_epoch"):
            lr_scheduler.on_end_epoch(ema_loss=self.ema_losses['train'].item())

        metrics = {}
        for loss_name, loss in self.losses.items():
            for k, v in loss.metrics.items():
                metric_name = f"{self.split}/{loss_name}/{k}"
                if log_to_wandb and hasattr(v, "as_wandb_metric"):
                    metrics[metric_name] = v.as_wandb_metric()
                elif hasattr(v, "compute"):
                    metrics[metric_name] = v.compute()
                else:
                    metrics[metric_name] = v

        if print_to_console:
            print_metrics_to_terminal(metrics)

        torch.set_grad_enabled(prev_state)

        return metrics
    
    def on_start_epoch(self):
        raise NotImplementedError

    def on_end_epoch(self):
        raise NotImplementedError

    def on_start_step(self, step_counter: int, split: str, batches_to_predict: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def on_end_step(self, step_counter: int, batches_to_predict: Dict[str, torch.Tensor]):
        raise NotImplementedError