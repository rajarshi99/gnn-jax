"""
Trying to reproduce DeepMind MeshGraphNet CylinderFlow example.
Node input: velocity (vx,vy) + node_type one-hot.
Target: change to next-step velocity (delta vx, delta vy).

TF is used only to decode the TFRecord into numpy arrays via trajectory_iterator_np.
All graph construction (senders/receivers), edge features, and training are JAX.
"""
import argparse
from pathlib import Path

from scripts.setup_run import setup_run
from scripts.get_model import get_model

from gnn_jax.data.cylinderflow_dm.train import train
from gnn_jax.data.cylinderflow_dm.evaluate import evaluate

import json

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--asymm", action="store_true")
    parser.add_argument("--tau", action="store_true")
    parser.add_argument("--dt_step", type=int, default=1, help="dt = dt_base (=0.01s) * dt_step")
    parser.add_argument("--zeroE", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    expt, cfg = setup_run(args)

    data_dir = Path(cfg["dataset"]["dir"])
    train_path = data_dir / cfg["dataset"]["train"]
    test_path = data_dir / cfg["dataset"]["test"]
    meta_path = data_dir / cfg["dataset"]["meta"]

    model = get_model(cfg, args.asymm, args.tau, expt)

    if args.tau:
        max_tstep = int(cfg[expt].get("max_tstep", 16))
        dt_step = args.dt_step
    else:
        max_tstep = None
        dt_step = None

    if args.custom:
        with open(cfg["custom"]["split_path"], "r") as f:
            split = json.load(f)

    if args.mode == "train":
        if args.custom:
            train_traj_ids = split["train_traj_ids"]
            train(model, cfg[expt],
                  train_path, meta_path, train_traj_ids=train_traj_ids,
                  max_tstep=max_tstep,
                  resume=cfg[expt].get("resume", False))
        else:
            train(model, cfg[expt],
                  train_path, meta_path,
                  max_tstep=max_tstep,
                  resume=cfg[expt].get("resume", False))

    elif args.mode == "eval":
        eval_dir = "eval"
        if args.zeroE:
            eval_dir += "_zeroE"
        cfg[expt]["eval_dir"] = str(Path(cfg[expt]["ckpt_dir"]) / eval_dir)
        if args.custom:
            test_traj_ids = split["test_traj_ids"]
            # train_path is intentional. The split is only on the train.tfrecord
            cfg[expt]["eval_dir"] += "_custom"
            evaluate(model, cfg[expt], train_path, meta_path, dt_step=dt_step, test_traj_ids=test_traj_ids, zeroE=args.zeroE, model_path=args.model)
        else:
            evaluate(model, cfg[expt], test_path, meta_path, dt_step=dt_step, zeroE=args.zeroE, model_path=args.model)

if __name__ == "__main__":
    main()

