import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import numpy as np
import hydra
from omegaconf import OmegaConf, open_dict
import wandb
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import hydra
import distinctipy
from utils.early_stopping import EarlyStopping
import random
import data.data_utils as data_utils
import logging
import itertools
import copy
import utils.table_creator as table_creator
import utils.nodaemonmultiproc as nodaemonmultiproc
import multiprocessing as mp
from models.backends.utils.spconv_utils import spconv
from data.dataloader_wrapper import DataLoaderCUDABus

@hydra.main(config_path="../conf", config_name="config_eva", version_base=None)
def main(_cfg):
    cfg = _cfg
    start_train(cfg)


def download_wandb_model_if_necessary(cfg):
    if cfg.checkpoint_path is None:
        return

    if cfg.checkpoint_path.startswith("wandb://"):
        logging.info("Downloading model from wandb to %s" % cfg.checkpoints_dir)
        # Download model from wandb and save to best_model.pth
        artifact = wandb.use_artifact(cfg.checkpoint_path.split("wandb://")[1], type="model")
        artifact.download(root=cfg.checkpoints_dir)
        cfg.checkpoint_path = os.path.join(cfg.checkpoints_dir, "best_model.pth")

def init_pytorch(cfg):
    torch.cuda.empty_cache() 
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    spconv.constants.SPCONV_ALLOW_TF32 = True

def start_train(cfg):
    if cfg.get("batch_exps", None) is not None:
        list_of_exps, list_of_overrides = derive_batch_configs(cfg)
        for i, exp_batch in enumerate(list_of_exps):
            overrides = list_of_overrides[i]
            batch_results = []
            metrics_to_get = None
            exp_batch_name = exp_batch["name"]
            exp_batch_cfgs = exp_batch["cfgs"]
            
            if cfg.batch_number_of_processes <= 1:
                for i, derived_cfg in enumerate(exp_batch_cfgs):
                    metrics_to_get = derived_cfg.table_metrics
                    logging.info(f"Batch of experiments named {exp_batch_name} - Running experiment {i+1}/{len(exp_batch_cfgs)} sequentially...")
                    batch_results.append(run_training(derived_cfg))
                    table_creator.create_table(Path(cfg.tables_path) / f"{exp_batch_name}.csv",
                                            batch_results,
                                            overrides,
                                            metrics_to_get,
                                            exp_batch_cfgs)
            else:
                pool = nodaemonmultiproc.NestablePool(min(cfg.batch_number_of_processes, len(exp_batch_cfgs)))
                results = pool.map(run_training, exp_batch_cfgs)
                pool.close()
                pool.join()
                
                metrics_to_get = exp_batch_cfgs[0].table_metrics
                table_creator.create_table(Path(cfg.tables_path) / f"{exp_batch_name}.csv",
                                           results,
                                           overrides,
                                           metrics_to_get,
                                           exp_batch_cfgs)
    else:
        result = run_training(cfg)
        metrics_to_get = cfg.table_metrics
        table_creator.create_table(Path(cfg.tables_path) / f"{cfg.exp_name}.csv", [result], [[]], metrics_to_get, [cfg])

def dict_to_dotlist(cfg, key_prefix=""):
    dotlist = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            dotlist += dict_to_dotlist(value, key_prefix=key_prefix + key + ".")
        else:
            dotlist.append((key_prefix + key, value))

    return dotlist

def derive_batch_configs(cfg):
    """
    Given a config that contains a list of possible experiment overrides, derive all possible configurations
    """

    base_cfg = cfg
    list_of_batch_exps = []
    list_of_overrides = []

    for batch_of_exp_overrides in base_cfg.batch_exps:
        cfgs = []
        # Iterate over all values of the batch_exps list
        overrides_not_unrolled = dict_to_dotlist(OmegaConf.to_container(batch_of_exp_overrides.params))

        grid_search_overrides = [x for x in overrides_not_unrolled if x[0].endswith("_grid")]
        const_overrides = [x for x in overrides_not_unrolled if not x[0].endswith("_grid")]

        # Go through all possible permutations of the grid search overrides
        grid_search_indices = list(itertools.product(*[list(range(len(x[1]))) for x in grid_search_overrides]))

        overrides_unrolled = []

        for index in grid_search_indices:
            derived_cfg = copy.deepcopy(base_cfg)
            changes_made = []

            for idx, (key, value) in enumerate(grid_search_overrides):
                changes_made.append((key[:-len("_grid")-1], value[index[idx]]))
                OmegaConf.update(derived_cfg, changes_made[-1][0], changes_made[-1][1])

            for key, value in const_overrides:
                changes_made.append((key, value))
                OmegaConf.update(derived_cfg, changes_made[-1][0], changes_made[-1][1])

            derived_cfg.exp_name += "_" + "_".join([str(x[0]) + "=" + str(x[1]) for x in changes_made])

            cfgs.append(derived_cfg)
            overrides_unrolled.append(changes_made)

        list_of_batch_exps.append({
            "name": batch_of_exp_overrides.name,
            "cfgs": cfgs
        })
        list_of_overrides.append(overrides_unrolled)

    return list_of_batch_exps, list_of_overrides

def setup_debug_modes(cfg, train_idxs, val_idxs, test_idxs):
    max_train_samples = None
    if cfg.debug == "one_sample" or cfg.debug == True:
        max_train_samples = 1
    elif cfg.debug == "two_samples":
        max_train_samples = 2
    elif cfg.debug == "five_samples":
        max_train_samples = 5
    elif cfg.debug == "ten_samples":
        max_train_samples = 10
    
    if max_train_samples is not None:
        # Pick max_train_samples random samples from the train_idxs with numpy
        # temporarily set a random seed
        prev_seed = np.random.get_state()
        from time import time
        np.random.seed(int(time()))
        train_idxs = np.random.choice(train_idxs, max_train_samples, replace=False)
        np.random.set_state(prev_seed)
        val_idxs = train_idxs[:]
        test_idxs = train_idxs[:]
        cfg.batch_size = min(cfg.batch_size, max_train_samples)
        cfg.data.num_workers = 0
    
    return train_idxs, val_idxs, test_idxs

def run_training(cfg):
    cfg = prepare_config(cfg)
    init_pytorch(cfg)

    checkpoints_dir = Path(cfg.checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Creating DS splits...")

    ct_numbers = data_utils.get_ct_numbers(cfg.data.dataset_path)
    train_idxs, val_idxs, test_idxs = data_utils.create_splits(ct_numbers, cfg.fold_idx)
    train_idxs, val_idxs, test_idxs = setup_debug_modes(cfg, train_idxs, val_idxs, test_idxs)

    if cfg.limit_num_samples:
        train_idxs = train_idxs[0:cfg.limit_num_samples]
        val_idxs = val_idxs[0:cfg.limit_num_samples]
        test_idxs = test_idxs[0:cfg.limit_num_samples]

    if cfg.eval.target_split == "val":
        test_idxs = val_idxs.copy()
    elif cfg.eval.target_split == "train":
        test_idxs = train_idxs.copy()

    with open_dict(cfg):
        cfg.train_idxs = train_idxs.tolist()
    
    train_ds = hydra.utils.instantiate(cfg.data, cfg=cfg, split='train', ct_numbers=ct_numbers[train_idxs], batch_size=cfg.batch_size, eval_only=cfg.eval.eval_only, _recursive_=False)
    val_ds = hydra.utils.instantiate(cfg.data, cfg=cfg, split='val', ct_numbers=ct_numbers[val_idxs], batch_size=cfg.batch_size, stats=train_ds.stats, eval_only=cfg.eval.eval_only, _recursive_=False)
    test_ds = hydra.utils.instantiate(cfg.data, cfg=cfg, split='test', ct_numbers=ct_numbers[test_idxs], batch_size=1, stats=train_ds.stats, _recursive_=False)

    if cfg.use_wandb:
        wandb.init(project="sparse-bones", entity="medcv3d",
                   name=cfg.exp_name, dir="/netscratch/martelleto",
                   config=OmegaConf.to_container(cfg),
                   id=cfg.wandb_id,
                   resume="allow")

    download_wandb_model_if_necessary(cfg)

    print("Instantiating net_runner...")
    net_runner = hydra.utils.instantiate(cfg.runners, cfg=cfg, _recursive_=False)

    if cfg.eval.eval_only:
        optimizer = None
        grad_scaler = None
        lr_scheduler = None
    else:
        optimizer = hydra.utils.instantiate(cfg.optim.optimizer, params=net_runner.parameters())
        grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer, _recursive_=False)
    
    epoch = 0

    if cfg.checkpoint_path is not None:
        chpt = torch.load(cfg.checkpoint_path)
        net_runner.load_state_dict(chpt['runner_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(chpt['optimizer_state_dict'])
        epoch = chpt['epoch']
        if lr_scheduler is not None:
            lr_scheduler.ctr = epoch * cfg.steps_per_epoch
            lr_scheduler.last_epoch = epoch

    print("Creating dataloaders...")
    train_dataloader = train_ds.to_dataloader(shuffle=not cfg.debug, steps_per_epoch=cfg.steps_per_epoch, enable_data_prefetch=cfg.enable_data_prefetch)
    val_dataloader = val_ds.to_dataloader(shuffle=False, steps_per_epoch=cfg.steps_per_validation, enable_data_prefetch=cfg.enable_data_prefetch)
    test_dataloader = test_ds.to_dataloader(shuffle=False, steps_per_epoch=None, enable_data_prefetch=False)

    logging.info("Train/val/test split: %d/%d/%d" % (len(train_ds), len(val_ds), len(test_ds)))

    best_model_pth = os.path.join(cfg.checkpoints_dir, "best_model.pth")
    early_stopping = EarlyStopping(patience=cfg.optim.early_stopping.patience, verbose=True, path=best_model_pth)

    best_val_metrics = None

    while epoch < cfg.max_epochs:
        ''' Adjust batch norm momentum '''
        logging.info('**** Epoch %d/%s ****' % (epoch + 1, cfg.max_epochs))
        
        if not cfg.eval.eval_only:
            train_metrics = net_runner.run_epoch("train", train_dataloader, optimizer, grad_scaler, lr_scheduler, epoch)

        state = {
            'epoch': epoch,
            'runner_state_dict': net_runner.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }

        if not cfg.train_only:
            val_metrics = net_runner.run_epoch("val", val_dataloader, None, None, None, epoch)
        else:
            val_metrics = train_metrics.copy()
        
        if cfg.eval.eval_only:
            best_val_metrics = val_metrics.copy()
            break

        # EA metric is to be minimized
        ea_metric = net_runner.ema_losses['val'] if not cfg.train_only else net_runner.ema_losses['train']
        if early_stopping(ea_metric, state, allow_counter=epoch * cfg.steps_per_epoch > cfg.min_steps):
            best_val_metrics = val_metrics.copy()
        
        if epoch % cfg.save_interval == 0 and epoch >= cfg.min_epochs_before_save:
            torch.save(state, os.path.join(cfg.checkpoints_dir, f"checkpoint_{epoch}.pth"))

        if wandb.run is not None:
            wandb.log({
                **train_metrics,
                **val_metrics,
                'lr': optimizer.param_groups[0]['lr']
            })

        if early_stopping.early_stop:
            logging.info(f"Early stopping with best score: {early_stopping.best_score:.5f}")
            if cfg.use_wandb:
                wandb.log({
                    "summary/ea_score": early_stopping.best_score
                })
            break

        epoch += 1

    if not cfg.eval.eval_only:
        final_state = {
            'epoch': epoch,
            'runner_state_dict': net_runner.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }
        torch.save(final_state,
                   os.path.join(cfg.checkpoints_dir, f"checkpoint_{epoch}_final.pth"))
    
    load_path = early_stopping.path if not cfg.eval.eval_only and cfg.max_epochs > 0 else cfg.checkpoint_path
    loaded = torch.load(load_path)
    net_runner.load_state_dict(loaded['runner_state_dict']) # Load best model found during training
    
    test_metrics = net_runner.run_epoch('test', test_dataloader, None, None, None, epoch)
    # Save test_metrics to file
    if cfg.use_wandb:
        wandb.log(test_metrics)

    wandb.finish()

    if best_val_metrics is None:
        return test_metrics
    else:
        return {
            **best_val_metrics,
            **test_metrics
        }

def prepare_config(cfg):
    assert cfg.debug in [None, True, False, "data", "one_sample", "two_samples", "five_samples", "ten_samples"], "Invalid debug mode"

    if cfg.get("root_path") is None:
        if Path(".ROOT_PATH").exists():
            with open(".ROOT_PATH", "r") as f:
                with open_dict(cfg):
                    cfg.root_path = f.read().strip()
    
    if cfg.get("root_path") is None:
        raise ValueError("Please specify the results root_path in the config file conf/config_eval.yaml or create a .ROOT_PATH file in the root of the project. The root_path should contain your dataset file (e.g. SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5).")

    with open_dict(cfg):
        if cfg.debug == "data":
            cfg.data.num_workers = 0
            cfg.batch_size = 5

        cfg.exp_name = "debug_" + cfg.exp_name if cfg.debug else cfg.exp_name

        if cfg.eval.eval_only:
            cfg.exp_name = "eval" + cfg.eval.target_split.capitalize() + "_" + cfg.exp_name

    with open_dict(cfg):
        timestr = datetime.datetime.now().strftime("%d.%m.%YY-%H:%M:%S")
        cfg.exp_name = cfg.exp_name + "_" + timestr + "_" + str(hash(OmegaConf.to_yaml(cfg, resolve=False)))
        cfg.exp_path = os.path.join(cfg.root_path, cfg.exp_name)
        Path(cfg.exp_path).mkdir(exist_ok=True, parents=True)

        # Converts ds_type to class_names, num_classes, class_colros and class_names fields
        class_encoder_map = data_utils.get_class_encoder_map("all_classes")
        cfg.data.class_encoder_map = class_encoder_map
        cfg.data.num_classes = len(class_encoder_map)

        colors = np.asarray(distinctipy.get_colors(cfg.data.num_classes)) * 255
        colors[0] = [0, 0, 0]
        cfg.data.class_colors = colors.tolist()

        # Sort class_encoder_map by value (ascending)
        cfg.data.class_names = [k for k, v in sorted(class_encoder_map.items(), key=lambda item: item[1])]
        assert len(cfg.data.class_names) == cfg.data.num_classes and cfg.data.class_names[0] == '00_other'
        
    return cfg

if __name__ == '__main__':
    main()