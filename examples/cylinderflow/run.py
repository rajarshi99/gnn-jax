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

    if args.mode == "train":
        train(model, cfg[expt], train_path, meta_path, max_tstep=max_tstep)
    elif args.mode == "eval":
        eval_dir = "eval"
        if args.zeroE:
            eval_dir += "_zeroE"
        cfg[expt]["eval_dir"] = str(Path(cfg[expt]["ckpt_dir"]) / eval_dir)
        evaluate(model, cfg[expt], test_path, meta_path, dt_step=dt_step)

if __name__ == "__main__":
    main()

