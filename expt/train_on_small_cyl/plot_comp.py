from pathlib import Path
import yaml

from scripts.plot_utils import *

# from itertools import combinations

out_dir = Path("output")

base_dir =  Path("/media/user/HDD-UT/rajarshi_data/cylinder_flow_expt/")
label_fname = "label.yaml"
train_fname = "train_logs.csv"
rollout_dirs = [
        "eval_final",
        "eval_final_zeroE",
        "test",
        "test_zeroE"
        ]
rollout_labels = [
        "On Unseen R",
        "On Unseen R with Zero Input",
        "Test Trajectories",
        "Test Trajectories with Zero Input"
        ]

plot_info_list = []
for run_dir in base_dir.iterdir():
    label_f = run_dir / label_fname
    if label_f.exists():
        with open(label_f, "r") as f:
            plot_info = yaml.safe_load(f).get("small_cyl")
        if plot_info is not None:
            print("Reading", run_dir)
            plot_info["run_dir"] = run_dir
            plot_info_list.append(plot_info)
        else:
            print("Skipping", run_dir)

# Plot loss values normal and smooth
# plot_loss(train_fname, plot_info_list, out_dir / "loss.pdf")
# plot_loss(train_fname, plot_info_list, out_dir / "loss_smooth.pdf", alpha=0.98)

# Plot rollout errors for some trajectories
# for traj_id in range(10):
#     for rollout_dir in rollout_dirs:
#         plot_rollout_traj(plot_info_list, rollout_dir, traj_id, out_dir / f"{rollout_dir}_{traj_id:04d}.pdf")

# Get statistics over all rollouts
for rollout_dir in rollout_dirs:
    plot_rollout_stats(plot_info_list, rollout_dir, out_dir / f"{rollout_dir}_stats_beg.pdf", lambda t: (t < 1))
    plot_rollout_stats(plot_info_list, rollout_dir, out_dir / f"{rollout_dir}_stats_mid.pdf", lambda t: (2.5 < t) & (t < 3.5))
    plot_rollout_stats(plot_info_list, rollout_dir, out_dir / f"{rollout_dir}_stats_end.pdf", lambda t: (t > 5))
