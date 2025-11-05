import wandb
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List


class MplColorHelper:
    def __init__(self, cmap_name: str, start_val: float, stop_val: float):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def resume_run(id):
    wandb.init(project="sparse-bonnet", entity="medcv3d", id=id, resume=True)
    wandb_dict = dict(wandb.config)
    return OmegaConf.create(wandb_dict)


def create_point_cloud_from_voxels(voxel_coords: np.ndarray,
                                   voxel_features: np.ndarray,
                                   colors: np.ndarray):
    pcloud = np.zeros((voxel_coords.shape[0], 3 + voxel_features.shape[1]), dtype=np.float32)
    pcloud[:, :3] = voxel_coords.astype(np.float32)
    pcloud[:, 3:] = voxel_features
    return create_point_cloud(pcloud, None, colors)

def create_point_cloud(points: np.ndarray,
                       target: np.ndarray,
                       colors: np.ndarray,
                       max_points=30000):
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
        if target is not None: target = target[indices]
    
    pcloud = np.zeros((points.shape[0], 6))
    pcloud[:, :3] = points[:, :3]

    if colors is not None and target is not None:
        pcloud[:, 3:] = colors[target.reshape(-1), :]
    elif points.shape[1] > 3:
        # Color by feature length if available
        feat_lengths = np.linalg.norm(points[:, 3:], axis=-1)
        # Replace invalid values with 0
        feat_lengths[~np.isfinite(feat_lengths)] = 0
        cmap = MplColorHelper('plasma', 0, 1)
        feat_lengths = 1.0 - (feat_lengths - np.min(feat_lengths)) / (np.max(feat_lengths) - np.min(feat_lengths))
        pcloud[:, 3:] = np.asarray(cmap.get_rgb(feat_lengths)[:, :3]) * 255
    else:
        # Set all points to white
        pcloud[:, 3:] = 255

    return wandb.Object3D(pcloud)


def calculate_cube_corners(center, size):
    cx, cy, cz = center
    sx, sy, sz = size / 2

    if np.linalg.norm(size) <= 1e-6:
        return None

    corners = [
        [cx + sx, cy + sy, cz + sz],
        [cx + sx, cy + sy, cz - sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz - sz],
        [cx - sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx - sx, cy - sy, cz - sz],
    ]

    corners = np.asarray(corners).tolist()
    return corners


def create_bboxes_vis(points: np.ndarray,
                      target: np.ndarray,
                      bboxes: np.ndarray,
                      bboxes_classes: np.ndarray,
                      class_names: List[str],
                      class_colors: np.ndarray,
                      max_boxes_to_vis=-1):
    boxes_to_vis = [
        (calculate_cube_corners(box[:3], box[3:]), cls) for box, cls in zip(bboxes, bboxes_classes) if cls != 0
    ]
    if len(boxes_to_vis) == 0:
        return None
    if max_boxes_to_vis > 0:
        boxes_to_vis = boxes_to_vis[0:max_boxes_to_vis]
    boxes_list = []

    pcloud = np.zeros((points.shape[0], 6))
    pcloud[:, :3] = points[:, :3]
    pcloud[:, 3:] = class_colors[target.reshape(-1), :]

    for i, (corners, cls) in enumerate(boxes_to_vis):
        if corners is not None:
            boxes_list.append(
                {
                    "corners": corners,
                    "label": str(class_names[int(cls)]),
                    "color": (np.asarray(class_colors[int(cls)]) / 255).tolist(),
                }
            )

    return wandb.Object3D.from_point_cloud(pcloud, boxes_list)


def create_conf_matrix(class_names: List[str],
                       cfmat: np.ndarray):
    plt.figure(figsize=(16, 16))

    df_distmat = pd.DataFrame(cfmat, columns=class_names, index=class_names) * 100
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(df_distmat, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f', annot_kws={'size': 6})
    plt.title('Confusion matrix')

    img = wandb.Image(plt.gcf())
    plt.close()
    return img


def scatter_plots(all_points_stats: list,
                  samples_dices: List[float],
                  log_prefix: str):
    scatters_dict = {}
    # First, concat all the stats in the list of dicts into a single dict
    all_points_stats = {k: np.concatenate([d[k] for d in all_points_stats]) for k in all_points_stats[0]}

    for k, v in all_points_stats.items():
        x_values = v
        y_values = [samples_dices[i][1] for i in range(len(samples_dices))]
        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        table = wandb.Table(data=data, columns=[k, "FAIR_Dice"])

        # Creates a wandb.scatter plot of the stats vs performance
        scatter = wandb.plot.scatter(table, k, "FAIR_Dice", title="FAIR_Dice (y) vs " + k + " (x)")
        scatters_dict[f"{log_prefix}/scatter_{k}"] = scatter

    return scatters_dict
