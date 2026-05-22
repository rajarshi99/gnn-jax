"""Diagnostic script for estimating cylinder center and radius from CylinderFlow meshes."""
import argparse
import yaml
from pathlib import Path

import numpy as np
from gnn_jax.data.cylinderflow_dm.load import trajectory_iterator_np, NodeType

import matplotlib.pyplot as plt

def get_R_errR(traj):
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
    return radius, radius_err

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)
data_info = cfg["dataset"]

data_dir_path = Path(data_info["dir"])
train_path = data_dir_path / data_info["train"]
test_path = data_dir_path / data_info["test"]
meta_path = data_dir_path / data_info["meta"]

r_CutOff = 0.05

# Plot histogram of train data
count_err = 0
count_small = 0
count_large = 0
r_train = []
trajectory_iterator_obj = trajectory_iterator_np(train_path, meta_path)
traj = next(trajectory_iterator_obj)
while traj is not None:
    radius, radius_err = get_R_errR(traj)
    if radius_err > 1e-5:
        count_err += 1
    if radius < r_CutOff:
        count_small += 1
    else:
        count_large += 1
    r_train.append(radius)
    traj = next(trajectory_iterator_obj, None)
print(f"Problem cases in training dataset {count_err}")

plt.hist(r_train, bins=20)
plt.axvline(r_CutOff, color="red", linestyle="--", label="R_c")
plt.text(
        (plt.xlim()[0] + r_CutOff)*0.5,
        plt.ylim()[1] * 0.2,
        f"Training Data\n{count_small} trajectories",
        ha="center"
        )
plt.text(
        (r_CutOff + plt.xlim()[1])*0.5,
        plt.ylim()[1] * 0.2,
        f"Unseen R\n{count_large} trajectories",
        ha="center"
        )
plt.xlabel("Cylinder Radius")
plt.ylabel("Trajectory Count")
plt.savefig("train_histo.png")
plt.close()

# Plot histogram of test data
count_err = 0
r_test = []
trajectory_iterator_obj = trajectory_iterator_np(test_path, meta_path)
traj = next(trajectory_iterator_obj)
while traj is not None:
    radius, radius_err = get_R_errR(traj)
    if radius_err > 1e-5:
        count_err += 1
    r_test.append(radius)
    traj = next(trajectory_iterator_obj, None)
print(f"Problem cases in test dataset {count_err}")

plt.hist(r_test, bins=20)
plt.axvline(r_CutOff, color="red", linestyle="--", label="R_c")
plt.xlabel("Cylinder Radius")
plt.ylabel("Trajectory Count")
plt.savefig("test_histo.png")
plt.close()
