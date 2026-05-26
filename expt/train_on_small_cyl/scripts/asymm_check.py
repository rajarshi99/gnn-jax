import jax
import numpy as np

from gnn_jax.meshgraphnet import load_checkpoint
from pathlib import Path

from gnn_jax.data.cylinderflow_dm.trajectory import Trajectory
from gnn_jax.data.cylinderflow_dm.load import threaded_trajectory_iterator, NodeType

def get_rev_edge_id(senders, receivers):
    edge_map = {
            (int(s), int(r)): i
            for i, (s,r) in enumerate(zip(senders, receivers))
            }

    rev_edge_id = np.full(len(senders), -1, dtype=np.int32)

    for i, (s,r) in enumerate(zip(senders, receivers)):
        rev_edge_id[i] = edge_map.get((int(r), int(s)), -1)

    return rev_edge_id

def asymm_check(model, cfg_eval, data_path, meta_path):
    rng = jax.random.PRNGKey(69)
    ckpt_dir = Path(cfg_eval["ckpt_dir"])
    state = load_checkpoint(ckpt_dir / "model_final")
    variables = {"params": state["params"], "stats": state["stats"]}

    traj_dict_it = threaded_trajectory_iterator(data_path, meta_path)
    for _ in range(10):
        traj_dict = next(traj_dict_it)
        traj = Trajectory(traj_dict, model.edge_feat_dim)

        # Gen random node_in
        rng, sub = jax.random.split(rng)
        node_in = jax.random.uniform(sub, shape=(traj.N,NodeType.SIZE+2))
        pred, intermediates = model.apply(
                variables, node_in, traj.edge_in, traj.senders, traj.receivers,
                mutable=["intermediates"]
                )

        rev_edge_id = get_rev_edge_id(traj.senders, traj.receivers)
        valid_mask = rev_edge_id != -1
        num_miss = np.sum(rev_edge_id == -1)

        print(num_miss, jax.tree_util.tree_map(lambda x: x.shape, intermediates))
        for layer_data in intermediates["intermediates"].values():
            m = layer_data["messages"][0]
            err = m[valid_mask] + m[rev_edge_id[valid_mask]]
            print(f"mean {err.mean()} | std {err.std()}")

