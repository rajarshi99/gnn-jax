import jax
import jax.numpy as jnp

from gnn_jax.meshgraphnet import load_checkpoint
from gnn_jax.data.cylinderflow_dm.load import threaded_trajectory_iterator
from gnn_jax.data.cylinderflow_dm.trajectory import Trajectory

from pathlib import Path
import csv
import time

def evaluate(model, cfg_eval, data_path, meta_path, test_traj_ids=None, zeroE=False):
    ckpt_dir = Path(cfg_eval["ckpt_dir"])
    state = load_checkpoint(ckpt_dir / "model_final")
    variables = {"params": state["params"], "stats": state["stats"]}

    eval_dir = Path(cfg_eval["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    traj_id = 0
    traj_dict_it = threaded_trajectory_iterator(data_path, meta_path, traj_ids=test_traj_ids)
    traj_dict = next(traj_dict_it)
    while traj_dict is not None:
        traj = Trajectory(traj_dict, model.edge_feat_dim)
        def step_fn(v_t,_):
            node_in = jnp.concatenate([v_t, traj.node_type_oh], axis=-1)
            pred_delta_v = model.apply(
                    variables, node_in, traj.edge_in, traj.senders, traj.receivers
                    )
            v_next = v_t + pred_delta_v
            return v_next, v_next

        if zeroE:
            v0 = jnp.zeros_like(traj.vel[0])
        else:
            v0 = traj.vel[0]

        _, v_seq = jax.lax.scan(step_fn, v0, length=traj.T-1)

        fout_path = eval_dir / f"traj_{traj_id:04d}.npz"
        if zeroE:
            jnp.savez(fout_path, pred=v_seq)
        else:
            jnp.savez(fout_path, pred=v_seq, gt=traj.vel[1:])
        print(f"traj_id {traj_id} | shape {v_seq.shape} | @ {fout_path}")
        traj_id += 1
        traj_dict = next(traj_dict_it)
