from gnn_jax.data.drivaerml.load_surface import DrivAerIterator

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

drivaer_iter = DrivAerIterator(cfg["dataset"])
for io_dict in drivaer_iter:
    for k,v in io_dict.items():
        print(k, type(v))

