import argparse
import yaml
from pathlib import Path

from gnn_jax.data.deepmind_cylinderflow import *

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

trajectory_iterator_obj = trajectory_iterator_np(train_path, meta_path)
for _ in range(10):
    traj = next(trajectory_iterator_obj)
    for k,v in traj.items():
        print(k, v.shape)

