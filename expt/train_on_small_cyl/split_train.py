"""Diagnostic script for estimating cylinder center and radius from CylinderFlow meshes."""
import argparse
import yaml
from pathlib import Path

import numpy as np
from gnn_jax.data.deepmind_cylinderflow import trajectory_iterator_np, NodeType

import json

parser = argparse.ArgumentParser(
        description="Example file parsing tf records of cylinderflow simulations"
        )
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)
data_info = cfg["dataset"]

data_dir_path = Path(data_info["dir"])
train_path = data_dir_path / data_info["train"]
meta_path = data_dir_path / data_info["meta"]

split_path = Path(cfg["custom"]["split_path"])
split = {
        "train_traj_ids": [],
        "test_traj_ids": []
        }

count = 0
r_CutOff = 0.05
count_s = 0
count_b = 0

count_err = 0
trajectory_iterator_obj = trajectory_iterator_np(train_path, meta_path)
for traj in trajectory_iterator_obj:
    num_nodes = traj["node_type"].shape[1]
    num_cells = traj["cells"].shape[1]

    node_type = traj["node_type"][0,:,0]
    mesh_pos = traj["mesh_pos"][0,:,:]
    wall_pos = mesh_pos[node_type == NodeType.WALL_BOUNDARY]
    y = wall_pos[:,1]
    y_min = y.min()
    y_max = y.max()
    tol = 0.01
    cylinder_mask = (y_min + tol < y) & (y < y_max - tol)
    cylinder_nodes = wall_pos[cylinder_mask]
    cylinder_center = cylinder_nodes.mean(axis=0)
    radii = np.linalg.norm(cylinder_nodes - cylinder_center[None,:], axis = 1)
    radius = radii.mean()
    radius_err = radii.std()
    print(f"{count:04d}")
    if radius_err > 1e-5:
        count_err += 1
    elif radius < r_CutOff:
        count_s += 1
        split["train_traj_ids"].append(count)
    else:
        count_b += 1
        split["test_traj_ids"].append(count)
    count += 1

print(f"Number of trajectories with cylinder radius less than {r_CutOff}: {count_s}")
print(f"Number of trajectories with cylinder radius more than {r_CutOff}: {count_b}")
print(f"Problem cases? {count_err}")

with open(split_path, "w") as f:
    json.dump(split, f, indent=2)
print(f"Saving split in {split_path}")

