from pathlib import Path
import yaml
import shutil
import time

def setup_run(args, expt_name):
    if args.mode == "train":
        if args.config is None:
            raise ValueError("--config required for training")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        run_dir = Path(cfg[expt_name]["out_dir"] + time.strftime("_%m%d_%H%M"))
        run_dir.mkdir(parents=True)
        print(f"Created dir {run_dir}")
        cfg[expt_name]["ckpt_dir"] = str(run_dir)
        cfg[expt_name]["eval_dir"] = str(run_dir / "eval_final")
        cfg[expt_name]["log"] = str(run_dir / "train_logs.csv")
        out_yaml = Path(run_dir) / "config.yaml"
        with open(out_yaml, "w") as  f:
            yaml.dump(cfg, f)
        shutil.copy(Path(__file__), run_dir / Path(__file__).name)

    else:
        if args.run_dir is None:
            raise ValueError("--run_dir required for evaluation")
        config_path = Path(args.run_dir) / "config.yaml"
        if not config_path.exists():
            raise ValueError("config.yaml not found")
        print(f"Fetching config from {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print(cfg)

    return cfg


