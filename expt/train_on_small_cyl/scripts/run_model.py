from pathlib import Path
import json

from gnn_jax.data.cylinderflow_dm.train import train
from gnn_jax.data.cylinderflow_dm.evaluate import evaluate
from scripts.asymm_check import asymm_check as check

def run_model(model, expt_name, args, cfg):
    data_dir = Path(cfg["dataset"]["dir"])
    train_path = data_dir / cfg["dataset"]["train"]
    meta_path = data_dir / cfg["dataset"]["meta"]

    split_path = Path(cfg["custom"]["split_path"])
    with open(split_path, "r") as f:
        split = json.load(f)

    if args.mode == "train":
        train_traj_ids = split["train_traj_ids"]
        train(model, cfg[expt_name], train_path, meta_path, train_traj_ids)
    elif args.mode == "eval":
        test_traj_ids = split["test_traj_ids"]
        evaluate(model, cfg[expt_name], train_path, meta_path, test_traj_ids)
    elif args.mode == "test":
        test_path = data_dir / cfg["dataset"]["test"]
        evaluate(model, cfg[expt_name], test_path, meta_path, zeroE=args.zeroE)
    elif args.mode == "check":
        test_path = data_dir / cfg["dataset"]["test"]
        check(model, cfg[expt_name], test_path, meta_path)

