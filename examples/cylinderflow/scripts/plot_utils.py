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
        err = traj["pred"]
    err = np.sqrt(np.mean(np.sum(err**2, axis=-1), axis=-1))
    if "t_skip" in traj:
        t_skip = traj["t_skip"]
    else:
        t_skip = 1
    t = np.arange(1, err.shape[0]+1) * t_skip * dt_min
    return t, err

def plot_loss(colors, labels, styles, train_paths, out_fname, alpha=None, init_ind=10):
    if alpha is None:
        plt.ylabel("Loss")
    else:
        plt.ylabel("Loss (smooth)")
    for c, lab, sty, train_path in zip(colors, labels, styles, train_paths):
        df = pd.read_csv(train_path)
        if len(df.index) < init_ind:
            init_ind = 0
        if alpha is None:
            plt.plot(df["step"][init_ind:], df["loss"][init_ind:],
                     alpha=0.69, color=c, label=lab, ls=sty)
        else:
            plt.plot(
                    df["step"][init_ind:],
                    ema(df["loss"], alpha=alpha)[init_ind:],
                    alpha=0.69, color=c, label=lab, ls=sty)

    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("Step")
    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()

# def plot_rollout_traj(plot_info_list, rollout_dir, traj_id, out_fname):
#     ls_list = ["-", "--", "-.", ":"]
#     for p in plot_info_list:
#         eval_dir = p["run_dir"] / rollout_dir
#         t_dirs = sorted(eval_dir.glob("dt_*"))
#         for i,t_dir in enumerate(t_dirs):
#             traj_f = eval_dir / t_dir / f"traj_id_{traj_id:04d}.npz"
#             t_skip_label = t_dir.name.split("_")[-1]
#             if not traj_f.exists():
#                 print(404, traj_f)
#                 continue
#             t, err = get_rollout_t_vs_err(traj_f)
#             label = p["label"]
#             if i > 0:
#                 label += rf" $\tau={int(t_skip_label)}\delta t$"
#             plt.plot(t, err, color=p["color"], ls=ls_list[i], label=label)
#     plt.legend()
#     plt.grid()
#     plt.yscale("log")
#     plt.xlabel("Time")
#     plt.ylabel("Error")
#     print(f"Saving @ {out_fname}")
#     plt.savefig(out_fname)
#     plt.close()

def plot_rollout_stats(colors, labels, traj_paths, out_fname, mask_fn):
    errors = []
    median_errors = []
    for i, t_dir in enumerate(traj_paths):
        e4trajs = []
        for traj_f in t_dir.glob("traj_*.npz"):
            t, err = get_rollout_t_vs_err(traj_f)
            e4trajs.append(err[mask_fn(t)].mean())
        errors.append(e4trajs)
        median_errors.append(np.median(e4trajs))
        print(labels[i], t_dir, np.min(e4trajs), np.max(e4trajs))

    sorted_idx = np.argsort(median_errors)

    labels_sorted = [labels[i] for i in sorted_idx[::-1]]
    errors_sorted = [errors[i] for i in sorted_idx[::-1]]
    colors_sorted = [colors[i] for i in sorted_idx[::-1]]

    bp = plt.boxplot(errors_sorted, patch_artist=True, vert=False)
    for patch, c in zip(bp["boxes"], colors_sorted):
        patch.set_facecolor(c)

    plt.yticks(range(1,len(labels_sorted)+1), labels_sorted)
    plt.xlabel("Time averaged RMSE")
    plt.xscale("log")

    plt.gca().set_axisbelow(True)
    plt.gca().grid(True, axis='y', linestyle='--', color='0.5', linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()

def plot_acc_cost(colors, labels, traj_paths, out_fname):
    for c, lab, t_dirs in zip(colors, labels, traj_paths):
        cost_list = []
        avg_err_list = []
        std_err_list = []
        for t_dir in t_dirs:
            n_tot = 0
            n_fail = 0
            err_list = []
            for i, traj_f in enumerate(t_dir.glob("traj_*.npz")):
                t, err = get_rollout_t_vs_err(traj_f)
                if np.all(np.isfinite(err)):
                    err_list.append(err[-1])
                else:
                    n_fail += 1
                n_tot += 1
            print("\t", lab, t_dir, "n_tot:", n_tot, "| n_fail:", n_fail)
            if n_fail > 0:
                continue
            cost_list.append(t.shape[0])
            avg_err_list.append(np.mean(err_list))
            std_err_list.append(np.std(err_list))

        plt.errorbar(cost_list, avg_err_list, yerr=std_err_list, fmt="o", capsize=4, color=c, label=lab)
        plt.errorbar(cost_list, avg_err_list, linestyle="--", color=c)

        # if len(cost_list) > cost_list_len:
        #     longest_cost_list = cost_list

    plt.xlabel("Cost: number of rollout iterations")
    # plt.xscale("log")
    plt.ylabel("End RMSE")
    plt.yscale("log")
    plt.legend()

    ax = plt.gca()

    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', color='0.5', linewidth=0.7, alpha=0.7)

    # secax = ax.twiny()
    # secax.set_xscale(ax.get_xscale())
    # secax.set_xlim(ax.get_xlim())
    # xticks = []
    # xticklabels = []
    # prev = None
    # for cost in longest_cost_list:
    #     val = dt_min * max(longest_cost_list) / cost
    #     if prev is None:
    #         xticks.append(cost)
    #         xticklabels.append(f"{(val):.2f}")
    #         prev = val
    #     elif np.log(val) - np.log(prev) > 0.2:
    #         xticks.append(cost)
    #         xticklabels.append(f"{(val):.2f}")
    #         prev = val
    # secax.set_xticks(xticks)
    # secax.set_xticklabels(xticklabels)
    # secax.set_xlabel("Time step value")
    # secax.xaxis.set_ticks_position("bottom")
    # secax.xaxis.set_label_position("bottom")
    # secax.spines["bottom"].set_position(("axes", -0.2))

    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()
