from gnn_jax.data.drivaerml.load_surface import DrivAerIterator
import jax

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config.yaml in train mode")
args = parser.parse_args()

rng = jax.random.key(42)

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)
base_dir = cfg["dataset"]["dir"]
data_cfg = cfg["dataset"]["cfg"]

drivaer_iter = iter(DrivAerIterator(rng, base_dir, data_cfg))
mesh, data = next(drivaer_iter)
print(mesh, data)

