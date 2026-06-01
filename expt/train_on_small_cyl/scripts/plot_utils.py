import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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
                label += f" {t_skip_label}"
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
    colors = {}

    for p in plot_info_list:
        eval_dir = p["run_dir"] / rollout_dir
        t_dirs = sorted(eval_dir.glob("dt_*"))

        for i,t_dir in enumerate(t_dirs):
            t_skip_label = t_dir.name.split("_")[-1]
            label = p["label"]
            if i > 0:
                label += f" {t_skip_label}"

            err_means[label] = []
            colors[label] = p["color"]
            print("\t", label, t_dir)
            for i, traj_f in enumerate(t_dir.glob("traj_*.npz")):
                t, err = get_rollout_t_vs_err(traj_f)
                err_means[label].append(err[mask_fn(t)].mean())

    x_lab = list(err_means.keys())
    y_val = [err_means[k] for k in x_lab]
    color = [colors[k] for k in x_lab]

    med = [np.median(y) for y in y_val]
    sorted_idx = np.argsort(med)

    x_lab_sorted = [x_lab[i] for i in sorted_idx]
    y_val_sorted = [y_val[i] for i in sorted_idx]
    color_sorted = [color[i] for i in sorted_idx]

    bp = plt.boxplot(y_val_sorted, patch_artist=True)
    for patch, c in zip(bp["boxes"], color_sorted):
        patch.set_facecolor(c)

    plt.xticks(range(1,len(x_lab_sorted)+1), x_lab_sorted, rotation=45)
    plt.ylabel("Mean rollout error")
    plt.yscale("log")
    plt.grid()

    plt.tight_layout()
    print(f"Saving @ {out_fname}")
    plt.savefig(out_fname)
    plt.close()
