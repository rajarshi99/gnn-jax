import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    # === Figure ===
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    # === Fonts ===
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # (optional but very good for papers)
    # "text.usetex": True,

    # === Lines ===
    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    # === Axes ===
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "0.5",
    "grid.alpha": 0.6,

    # === Ticks ===
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # === Legend ===
    "legend.frameon": False,

    # === Colors (clean default cycle) ===
    "axes.prop_cycle": mpl.cycler(color=[
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
    ]),

    # === Spines ===
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# From meta.json
dt_min = 0.01

def ema(y, alpha=0.8):
    y_smooth = []
    y_prev = y[0]
    for val in y:
        y_prev = alpha*y_prev + (1 - alpha)*val
        y_smooth.append(y_prev)
    return np.array(y_smooth)

def get_rollout_t_vs_err(traj_rollout_path):
    traj = np.load(traj_rollout_path)
    if "gt" in traj:
        err = traj["pred"] - traj["gt"]
    else:
        err = traj["pred"] # zeroE case
    if "t_skip" in traj:
        t_skip = traj["t_skip"]
    else:
        t_skip = 1
    err = np.linalg.norm(err, axis=-1)
    err = err.mean(axis=-1) # Average over space
    t = np.arange(1, err.shape[0]+1) * t_skip * dt_min
    return t, err

def plot_loss(train_fname, plot_info_list, out_fname, alpha=None, init_ind=10):
    if alpha is None:
        plt.ylabel("Loss")
    else:
        plt.ylabel("Loss (smooth)")
    for p in plot_info_list:
        df = pd.read_csv(p["run_dir"] / train_fname)
        if len(df.index) < init_ind:
            init_ind = 0
        if alpha is None:
            plt.plot(df["step"][init_ind:], df["loss"][init_ind:],
                     alpha=0.69, color=p["color"], label=p["label"])
        else:
            plt.plot(
                    df["step"][init_ind:],
                    ema(df["loss"], alpha=alpha)[init_ind:],
                    alpha=0.69, color=p["color"], label=p["label"])
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("Step")
    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()

def plot_rollout_traj(plot_info_list, rollout_dir, traj_id, out_fname):
    ls_list = ["-", "--", "-.", ":"]
    for p in plot_info_list:
        eval_dir = p["run_dir"] / rollout_dir
        t_dirs = sorted(eval_dir.glob("dt_*"))
        for i,t_dir in enumerate(t_dirs):
            traj_f = eval_dir / t_dir / f"traj_id_{traj_id:04d}.npz"
            t_skip_label = t_dir.name.split("_")[-1]
            if not traj_f.exists():
                print(404, traj_f)
                continue
            t, err = get_rollout_t_vs_err(traj_f)
            label = p["label"]
            if i > 0:
                label += rf" $\tau={int(t_skip_label)}\delta t$"
            plt.plot(t, err, color=p["color"], ls=ls_list[i], label=label)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Error")
    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()

def plot_rollout_stats(plot_info_list, rollout_dir, out_fname, mask_fn):
    err_means = {}
    color_dict = {}

    for p in plot_info_list:
        eval_dir = p["run_dir"] / rollout_dir
        t_dirs = sorted(eval_dir.glob("dt_*"))

        for i,t_dir in enumerate(t_dirs):
            t_skip_label = t_dir.name.split("_")[-1]
            label = p["label"]
            if i > 0:
                label += rf" $\tau={int(t_skip_label)}\delta t$"

            err_means[label] = []
            color_dict[label] = p["color"]
            print("\t", label, t_dir)
            for i, traj_f in enumerate(t_dir.glob("traj_*.npz")):
                t, err = get_rollout_t_vs_err(traj_f)
                err_means[label].append(err[mask_fn(t)].mean())

    labels = list(err_means.keys())
    errors = [err_means[k] for k in labels]
    colors = [color_dict[k] for k in labels]

    med = [np.median(e4trajs) for e4trajs in errors]
    sorted_idx = np.argsort(med)

    labels_sorted = [labels[i] for i in sorted_idx[::-1]]
    errors_sorted = [errors[i] for i in sorted_idx[::-1]]
    colors_sorted = [colors[i] for i in sorted_idx[::-1]]

    bp = plt.boxplot(errors_sorted, patch_artist=True, vert=False)
    for patch, c in zip(bp["boxes"], colors_sorted):
        patch.set_facecolor(c)

    plt.yticks(range(1,len(labels_sorted)+1), labels_sorted)
    plt.xlabel("Mean rollout error")
    plt.xscale("log")

    plt.gca().set_axisbelow(True)
    plt.gca().grid(True, axis='y', linestyle='--', color='0.5', linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()

def plot_acc_cost(plot_info_list, rollout_dir, out_fname, mask_fn):
    for p in plot_info_list:
        label = p["label"]
        eval_dir = p["run_dir"] / rollout_dir


        cost_list = []
        avg_err_list = []
        std_err_list = []

        t_dirs = sorted(eval_dir.glob("dt_*"))
        for i,t_dir in enumerate(t_dirs):
            n_tot = 0
            n_fail = 0
            tavg_err_list = []
            for i, traj_f in enumerate(t_dir.glob("traj_*.npz")):
                t, err = get_rollout_t_vs_err(traj_f)
                if np.all(np.isfinite(err)):
                    tavg_err = err[mask_fn(t)].mean()
                    tavg_err_list.append(tavg_err)
                else:
                    n_fail += 1
                n_tot += 1
            print("\t", label, t_dir, "n_tot:", n_tot, "| n_fail:", n_fail)
            cost_list.append(t.shape[0])
            avg_err_list.append(np.mean(tavg_err_list))
            std_err_list.append(np.std(tavg_err_list))

        plt.errorbar(cost_list, avg_err_list, yerr=std_err_list, fmt="o", capsize=4, color=p["color"], label=label)
        plt.errorbar(cost_list, avg_err_list, linestyle="--", color=p["color"])

    plt.xlabel("Cost: No. Rollout Iterations")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()

    plt.gca().set_axisbelow(True)
    plt.gca().grid(True, linestyle='--', color='0.5', linewidth=0.7, alpha=0.7)

    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()
