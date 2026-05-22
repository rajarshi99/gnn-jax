import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import yaml
import jax.numpy as jnp

from itertools import combinations

out_dir = Path("output")

base_dir =  Path("/media/user/HDD-UT/rajarshi_data/cylinder_flow_expt/")
label_fname = "label.yaml"
rollout_dir = "eval_test_zeroE"

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

# Plot rollout errors for some trajectories
def get_rollout_err(traj_rollout_path):
    traj = jnp.load(traj_rollout_path)
    v = traj["pred"]
    E = jnp.sum(v*v, axis=-1)
    meanE = jnp.mean(E, axis=-1)
    return meanE

for traj_id in range(10):
    for plot_info in plot_info_list:
        err = get_rollout_err(plot_info["run_dir"] / rollout_dir / f"traj_{traj_id:04d}.npz")
        plt.plot(err, color=plot_info["color"], label=plot_info["label"])
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Energy")
    plt.savefig(out_dir / f"zeroE_{traj_id:04d}.png")
    plt.close()

# Get statistics over all rollouts
comp_id = 0
comp_list = []
for run1, run2 in combinations(plot_info_list, 2):
    label1 = run1["label"]
    label2 = run2["label"]

    counts = [0]*4
    traj_files1 = {f.name for f in (run1["run_dir"] / rollout_dir).glob("traj_*.npz")}
    traj_files2 = {f.name for f in (run2["run_dir"] / rollout_dir).glob("traj_*.npz")}
    for traj_f in (traj_files1 & traj_files2): # Set intersection
        print(f"{comp_id} | {traj_f} | {rollout_dir}")
        err1 = get_rollout_err(run1["run_dir"] / rollout_dir / traj_f)
        err2 = get_rollout_err(run2["run_dir"] / rollout_dir / traj_f)
        diff = err1 < err2
        if jnp.all(diff):
            counts[0] += 1
        elif not jnp.any(diff):
            counts[1] += 1
        elif diff[0]:
            counts[2] += 1
        else:
            counts[3] += 1

    bin_names = [f"{label1}\nbetter", f"{label2}\nbetter", f"{label1}\nbetter\ninitially", f"{label2}\nbetter\ninitially"]
    colors = [run1["color"], run2["color"]] + 2*["black"]
    total = sum(counts)
    pct_vals = [100*v/total for v in counts]
    plt.title(f"Evaluation on: Test data with 0 input ({total} trajectories)")
    plt.ylabel("Percentage of trajectories")
    plt.bar(bin_names, pct_vals, color=colors)
    plt.savefig(out_dir / f"zeroE_bar_comp{comp_id}.png")
    plt.close()
    print(comp_id, counts)
    
    comp = {
            "data": "test zeroE",
            "model_a": label1,
            "model_b": label2,
            "counts": {
                "a_better": counts[0],
                "b_better": counts[1],
                "a_to_b": counts[2],
                "b_to_a": counts[3],
                }
            }
    comp_list.append(comp)
    comp_id += 1

with open(out_dir / "zeroE_comparison.yaml", "w") as f:
    yaml.dump(comp_list, f)

