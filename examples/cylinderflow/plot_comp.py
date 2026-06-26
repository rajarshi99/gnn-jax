import argparse
from pathlib import Path
import yaml

from scripts.plot_utils import *

# from itertools import combinations

out_dir = Path("output")

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True)
args = parser.parse_args()

base_dir =  Path(args.base)
label_fname = "label.yaml"
train_fname = "train_logs.csv"

rollout_desc = {
        "eval"              :  "Test Trajectories",
        "eval_zeroE"        :  "Test Trajectories with Zero Input",
        "eval_custom"       :  "On Unseen R",
        "eval_zeroE_custom" :  "On Unseen R with Zero Input",
        }

class Runs:
    def __init__(self):
        self.run_paths = []
        self.info_dicts = []

    def add(self, run_path, info_dict):
        self.run_paths.append(Path(run_path))
        self.info_dicts.append(info_dict)
        print(f"Reading run dir {run_path}")
        for k,v in info_dict.items():
            if k in rollout_desc:
                print(f" {k}: {rollout_desc[k]}")
            else:
                print(f" {k}: {v}")

    def get_train_info(self):
        colors = []
        labels = []
        styles = []
        train_paths = []
        for run_path, info_dict in zip(self.run_paths, self.info_dicts):
            colors.append(info_dict["color"])
            labels.append(info_dict["label"])
            styles.append(info_dict["style"])
            train_paths.append(run_path / train_fname)
        return colors, labels, styles, train_paths

    def get_rollout_info(self, key, t_skip_list=None):
        colors = []
        labels = []
        traj_paths = []
        for run_path, info_dict in zip(self.run_paths, self.info_dicts):
            if key in info_dict:
                if info_dict[key] is True:
                    color = info_dict["color"]
                    label = info_dict["label"]
                else:
                    color = info_dict[key].get("color", "brown")
                    label = info_dict[key].get("label", "?")

                if t_skip_list is None:
                    traj_path = list((run_path / key).glob("dt_*"))
                    if len(traj_path) > 1:
                        traj_paths.append(traj_path)
                        colors.append(color)
                        labels.append(label)

                else:
                    for t_skip in t_skip_list:
                        traj_path = run_path / key / f"dt_{t_skip:02d}"
                        if traj_path.exists():
                            colors.append(color)
                            if t_skip == 1:
                                labels.append(label)
                            else:
                                labels.append(
                                        label + rf" $\tau={t_skip}\delta t$"
                                        )
                            traj_paths.append(traj_path)
        return colors, labels, traj_paths

# Load run info
runs = Runs()
for run_path in base_dir.iterdir():
    label_path = run_path / label_fname
    if label_path.exists():
        with open(label_path, "r") as f:
            info_dict = yaml.safe_load(f)
        runs.add(run_path, info_dict)

# Plot loss values normal and smooth
colors, labels, styles, train_paths = runs.get_train_info()
plot_loss(colors, labels, styles, train_paths, out_dir / "loss.pdf")
plot_loss(colors, labels, styles, train_paths, out_dir / "loss_smooth.pdf", alpha=0.98)

# Get statistics over all rollouts
for key in rollout_desc:
    print(f"Begin rollout {key}")
    colors, labels, traj_paths = runs.get_rollout_info(key, [1, 4, 8, 10])
    plot_rollout_stats(colors, labels, traj_paths, out_dir / f"{key}_stats_beg.pdf", lambda t: (t < 1))
    plot_rollout_stats(colors, labels, traj_paths, out_dir / f"{key}_stats_mid.pdf", lambda t: (2.5 < t) & (t < 3.5))
    plot_rollout_stats(colors, labels, traj_paths, out_dir / f"{key}_stats_end.pdf", lambda t: (t > 5))

    colors, labels, traj_paths = runs.get_rollout_info(key)
    plot_acc_cost(colors, labels, traj_paths, out_dir / f"{key}_acc_cost.pdf")
    print("_"*80)
