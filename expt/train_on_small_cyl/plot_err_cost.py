from pathlib import Path
import yaml

from scripts.plot_utils import *

# from itertools import combinations

out_dir = Path("output")

base_dir =  Path("/media/user/HDD-UT/rajarshi_data/cylinder_flow_expt/")
label_fname = "label.yaml"
train_fname = "train_logs.csv"
rollout_dirs = [
        "eval",
        "eval_zeroE",
        ]
rollout_labels = [
        "Test Trajectories",
        "Test Trajectories with Zero Input"
        ]

plot_info_list = []
for run_dir in base_dir.iterdir():
    label_f = run_dir / label_fname
    if label_f.exists():
        with open(label_f, "r") as f:
            plot_info = yaml.safe_load(f).get("err_cost")
        if plot_info is not None:
            print("Reading", run_dir)
            plot_info["run_dir"] = run_dir
            plot_info_list.append(plot_info)
        else:
            print("Skipping", run_dir)

for rollout_dir in rollout_dirs:
    plot_acc_cost(plot_info_list, rollout_dir, out_dir / f"err_cost_{rollout_dir}.pdf", lambda t: (t > 5))
