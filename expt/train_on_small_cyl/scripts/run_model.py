from pathlib import Path
import json

from gnn_jax.data.cylinderflow_dm.train import train
from gnn_jax.data.cylinderflow_dm.evaluate import evaluate
from scripts.asymm_check import asymm_check as check

def run_model(model, expt_name, args, cfg, dt_flag=False):
    data_dir = Path(cfg["dataset"]["dir"])
    train_path = data_dir / cfg["dataset"]["train"]
    meta_path = data_dir / cfg["dataset"]["meta"]

    split_path = Path(cfg["custom"]["split_path"])
    with open(split_path, "r") as f:
        split = json.load(f)

    max_tstep = None
    dt_step = None
    if dt_flag:
        if args.mode == "train":
            max_tstep = int(cfg[expt_name].get("max_tstep"))
        else:
            dt_step = args.dt_step

    run_dir = Path(args.run_dir)
    if args.mode == "train":
        train_traj_ids = split["train_traj_ids"]
        train(model, cfg[expt_name], train_path, meta_path, max_tstep=max_tstep, train_traj_ids=train_traj_ids)
    elif args.mode == "eval":
        test_traj_ids = split["test_traj_ids"]
        cfg[expt_name]["eval_dir"] = str(run_dir / "eval_final")
        evaluate(model, cfg[expt_name], train_path, meta_path, dt_step=dt_step, test_traj_ids=test_traj_ids)
    elif args.mode == "test":
        test_path = data_dir / cfg["dataset"]["test"]
        out_dir_name = "test"
        if args.zeroE:
            out_dir_name += "_zeroE"
        cfg[expt_name]["eval_dir"] = str(run_dir / out_dir_name)
        evaluate(model, cfg[expt_name], test_path, meta_path, dt_step=dt_step, zeroE=args.zeroE)
    elif args.mode == "check":
        test_path = data_dir / cfg["dataset"]["test"]
        check(model, cfg[expt_name], test_path, meta_path)

