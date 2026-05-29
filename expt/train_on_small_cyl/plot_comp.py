from pathlib import Path
import yaml

from scripts.plot_utils import *

# from itertools import combinations

out_dir = Path("output")

base_dir =  Path("/media/user/HDD-UT/rajarshi_data/cylinder_flow_expt/")
label_fname = "label.yaml"
train_fname = "train_logs.csv"
rollout_dirs = {
        "On Unseen R": "eval_final",
        "Test Trajectories": "test",
        "Test Trajectories with Zero Input": "test_zeroE"
        }

plot_info_list = []
for run_dir in base_dir.iterdir():
    label_f = run_dir / label_fname
    if label_f.exists():
        with open(label_f, "r") as f:
            plot_info = yaml.safe_load(f)
    else:
        plot_info = {
                "label": run_dir.name,
                "color": "black"
                }
    plot_info["run_dir"] = run_dir
    plot_info_list.append(plot_info)

# Plot loss values
plot_loss(train_fname, plot_info_list, out_dir / "loss.png")

# Plot loss values smooth using exp moving avg
plot_loss(train_fname, plot_info_list, out_dir / "loss_smooth.png", alpha=0.98)

# Plot rollout errors for some trajectories
for traj_id in range(10):
    plot_rollout_traj(plot_info_list, rollout_dirs["Test Trajectories"], traj_id, out_dir / f"test_traj_{traj_id:04d}")
    plot_rollout_traj(plot_info_list, rollout_dirs["On Unseen R"], traj_id, out_dir / f"unseenR_traj_{traj_id:04d}")

# # Get statistics over all rollouts
# comp_id = 0
# comp_list = []
# for data_label,rollout_dir in rollout_dirs.items():
#     for run1, run2 in combinations(plot_info_list, 2):
#         label1 = run1["label"]
#         label2 = run2["label"]
#
#         counts = [0]*4
#         traj_files1 = {f.name for f in (run1["run_dir"] / rollout_dir).glob("traj_*.npz")}
#         traj_files2 = {f.name for f in (run2["run_dir"] / rollout_dir).glob("traj_*.npz")}
#         for traj_f in (traj_files1 & traj_files2): # Set intersection
#             print(f"{comp_id} | {traj_f} | {data_label} | {rollout_dir}")
#             err1 = get_rollout_err(run1["run_dir"] / rollout_dir / traj_f)
#             err2 = get_rollout_err(run2["run_dir"] / rollout_dir / traj_f)
#             diff = err1 < err2
#             if jnp.all(diff):
#                 counts[0] += 1
#             elif not jnp.any(diff):
#                 counts[1] += 1
#             elif diff[0]:
#                 counts[2] += 1
#             else:
#                 counts[3] += 1
#
#         bin_names = [f"{label1}\nbetter", f"{label2}\nbetter", f"{label1}\nbetter\ninitially", f"{label2}\nbetter\ninitially"]
#         colors = [run1["color"], run2["color"]] + 2*["black"]
#         total = sum(counts)
#         pct_vals = [100*v/total for v in counts]
#         plt.title(f"Evaluation on: {data_label} ({total} trajectories)")
#         plt.ylabel("Percentage of trajectories")
#         plt.bar(bin_names, pct_vals, color=colors)
#         plt.savefig(out_dir / f"bar_comp{comp_id}.png")
#         plt.close()
#         print(comp_id, counts)
#
#         comp = {
#                 "data": data_label,
#                 "model_a": label1,
#                 "model_b": label2,
#                 "counts": {
#                     "a_better": counts[0],
#                     "b_better": counts[1],
#                     "a_to_b": counts[2],
#                     "b_to_a": counts[3],
#                     }
#                 }
#         comp_list.append(comp)
#         comp_id += 1
#
# with open(out_dir / "comparison.yaml", "w") as f:
#     yaml.dump(comp_list, f)
#
