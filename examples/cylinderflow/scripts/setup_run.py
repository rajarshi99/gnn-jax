from pathlib import Path
import yaml
import shutil
import time

def setup_run(args):
    expt_dict = {
            (False,False): "base",
            (False,True): "tau",
            (True,False): "asymm",
            (True,True): "asymm_tau",
            }
    expt = expt_dict[(args.asymm, args.tau)]

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        if args.config is None:
            raise ValueError("--config required for training")

        existing_ckpt_dir = cfg[expt].get("ckpt_dir", None)
        if existing_ckpt_dir is not None and Path(existing_ckpt_dir).exists():
            ckpt_dir = Path(existing_ckpt_dir)
            model_dirs = [d for d in ckpt_dir.glob("model_*") if d.is_dir()]
            if len(model_dirs) > 0:
                last_model_dir = max(model_dirs, key=lambda d: int(d.name.split("_")[-1]))
                cfg[expt]["resume"] = str(ckpt_dir / last_model_dir)
                print(f"Training from {last_model_dir}")
                return expt, cfg

        run_dir = Path(cfg[expt]["out_dir"] + time.strftime("_%m%d_%H%M"))
        run_dir.mkdir(parents=True)
        print(f"Created dir {run_dir}")
        cfg[expt]["ckpt_dir"] = str(run_dir)
        cfg[expt]["log"] = str(run_dir / "train_logs.csv")
        out_yaml = Path(run_dir) / "config.yaml"
        with open(out_yaml, "w") as  f:
            yaml.dump(cfg, f)
        shutil.copy(Path(__file__), run_dir / Path(__file__).name)
        return expt, cfg

    if args.custom:
        cfg["custom"]["flag"] = True

    return expt, cfg


