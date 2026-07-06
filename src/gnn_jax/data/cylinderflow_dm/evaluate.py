import jax
import jax.numpy as jnp

from gnn_jax.meshgraphnet import load_checkpoint
from gnn_jax.data.cylinderflow_dm.load import threaded_trajectory_iterator
from gnn_jax.data.cylinderflow_dm.trajectory import Trajectory

from pathlib import Path
import csv
import time
import json

def evaluate(model, cfg_eval, data_path, meta_path, dt_step=None, test_traj_ids=None, zeroE=False, model_path=None):
    ckpt_dir = Path(cfg_eval["ckpt_dir"])
    if model_path is None:
        state = load_checkpoint(ckpt_dir / "model_final")
    else:
        state = load_checkpoint(model_path)
    variables = {"params": state["params"], "stats": state["stats"]}

    with open(meta_path, "r") as f:
        meta = json.load(f)
    dt_min = float(meta["dt"])

    if dt_step is None:
        t_skip = 1
        dt_phy = dt_min
    else:
        t_skip = dt_step
        dt_phy = dt_min * dt_step

    eval_dir = cfg_eval.get("eval_dir")
    if eval_dir is None:
        eval_dir = Path(cfg_eval["ckpt_dir"]) / "eval"
    else:
        eval_dir = Path(eval_dir)
    eval_dir = eval_dir / f"dt_{t_skip:02d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    def rollout(v0, node_type_oh, edge_in, senders, receivers, mask, num_steps):
        if dt_step is None:
            def step_fn(v_t,_):
                node_in = jnp.concatenate([v_t, node_type_oh], axis=-1)
                pred = model.apply(
                        variables, node_in, edge_in, senders, receivers
                        )
                delta_v = model.apply(
                        variables,
                        pred,
                        method=lambda m,x: m.out_data_norm.denormalize(x)
                        )
                v_temp = v_t + delta_v
                v_next = jnp.where(mask[:,None], v_temp, v_t)
                return v_next, v_next
        else:
            def step_fn(v_t,_):
                node_in = jnp.concatenate([v_t, node_type_oh], axis=-1)
                pred = model.apply(
                        variables, dt_phy, node_in, edge_in, senders, receivers
                        )
                delta_v = model.apply(
                        variables,
                        pred,
                        method=lambda m,x: m.out_data_norm.denormalize(x)
                        )
                v_temp = v_t + delta_v
                v_next = jnp.where(mask[:,None], v_temp, v_t)
                return v_next, v_next
        return jax.lax.scan(step_fn, v0, length=num_steps)[1]
    # rollout = jax.jit(rollout)

    traj_id = 0
    traj_dict_it = threaded_trajectory_iterator(data_path, meta_path, traj_ids=test_traj_ids)
    traj_dict = next(traj_dict_it)
    while traj_dict is not None:
        traj = Trajectory(traj_dict, model.edge_feat_dim)
        if zeroE:
            v0 = jnp.zeros_like(traj.vel[0])
        else:
            v0 = traj.vel[0]
        # num_steps = int((traj.T - 1) // t_skip)
        num_steps = (traj.T - 1) // t_skip
        v_seq = rollout(v0, traj.node_type_oh, traj.edge_in, traj.senders, traj.receivers, traj.mask, num_steps)

        v_gt = traj.vel[t_skip::t_skip]
        fout_path = eval_dir / f"traj_id_{traj_id:04d}.npz"
        if zeroE:
            jnp.savez(fout_path, t_skip=t_skip, pred=v_seq)
        else:
            jnp.savez(fout_path, t_skip=t_skip, pred=v_seq, gt=v_gt)

        print(f"traj_id {traj_id} | shape {v_seq.shape} | @ {fout_path} | {v_seq.shape} | {v_gt.shape}")
        traj_id += 1
        traj_dict = next(traj_dict_it, None)
