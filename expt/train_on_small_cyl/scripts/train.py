import jax
import jax.numpy as jnp

import optax
from flax import linen as nn

import jraph
from gnn_jax.meshgraphnet import MeshGraphNet, save_checkpoint, close_checkpointer
from gnn_jax.data.deepmind_cylinderflow import threaded_trajectory_iterator, NodeType

from pathlib import Path
import time
import csv

# -------------------------
# Utilities
# -------------------------

def next_pow2(n: int):
    if n <= 1:
        return 1
    return 1 << (n-1).bit_length()

def create_variables(rng, model):
    # dummy init (only feature dims matter)
    N = 4
    E = 6
    node_dim = model.node_feat_dim
    node_in = jnp.zeros((N, node_dim), dtype=jnp.float32)
    edge_in = jnp.zeros((E, model.edge_feat_dim), dtype=jnp.float32)
    senders = jnp.zeros((E,), dtype=jnp.int32)
    receivers = jnp.zeros((E,), dtype=jnp.int32)

    return model.init(rng, node_in, edge_in, senders, receivers)

class Trajectory:
    def __init__(self, traj, edge_feat_dim=3):
        # Convert required arrays to JAX
        mesh_pos = jnp.asarray(traj["mesh_pos"][0], dtype=jnp.float32)            # (N,2)
        node_type = jnp.asarray(traj["node_type"][0, :, 0], dtype=jnp.int32)      # (N,)
        self.vel = jnp.asarray(traj["velocity"], dtype=jnp.float32)                    # (T,N,2)
        self.senders = jnp.asarray(traj["senders"], dtype=jnp.int32)                   # (E,)
        self.receivers = jnp.asarray(traj["receivers"], dtype=jnp.int32)               # (E,)

        self.N = mesh_pos.shape[0]
        self.T = self.vel.shape[0]
        self.E = self.senders.shape[0]

        # --- Edge features: (dx, dy, dist) or (dist) ---
        rel = mesh_pos[self.receivers] - mesh_pos[self.senders]              # (E,2)
        dist = jnp.linalg.norm(rel, axis=1, keepdims=True)                             # (E,1)
        if edge_feat_dim == 1:
            self.edge_in = dist
        elif edge_feat_dim == 2:
            self.edge_in = rel
        else:
            self.edge_in = jnp.concatenate([rel, dist], axis=1).astype(jnp.float32)    # (E,3)

        # --- Node type one-hot ---
        self.node_type_oh = jax.nn.one_hot(node_type, num_classes=NodeType.SIZE, dtype=jnp.float32)  # (N,num_types)

        # --- Loss mask: only NORMAL and OUTFLOW nodes ---
        self.mask = (
                (node_type == NodeType.NORMAL) |
                (node_type == NodeType.OUTFLOW)
                ).astype(jnp.float32)        # (N,)

    def get_graph(self):
        node_in = jnp.concatenate([self.vel[0], self.node_type_oh], axis=-1)
        return jraph.GraphsTuple(
            nodes=node_in,               # (N, F)
            edges=self.edge_in,          # (E, D)
            senders=self.senders,        # (E,)
            receivers=self.receivers,    # (E,)
            n_node=jnp.array([node_in.shape[0]]),
            n_edge=jnp.array([self.edge_in.shape[0]]),
            globals=None,
        )

    def get_random_data_node_in_out(self, rng):
        rng, sub = jax.random.split(rng)
        t = int(jax.random.randint(sub, (), 0, self.T - 1))
        v_t = (self.vel[t])          # (N,2)
        v_t1 = (self.vel[t+1])       # (N,2)
        target_delta_v = v_t1 - v_t
        node_in = jnp.concatenate([v_t, self.node_type_oh], axis=-1)
        return rng, node_in, target_delta_v


def train(model, cfg_train, train_path, meta_path, train_traj_ids=None):
    seed = int(cfg_train.get("seed", 0))
    lr = float(cfg_train.get("learning_rate", 1e-4))
    steps = int(cfg_train.get("steps", 500))
    steps_per_log = int(cfg_train.get("steps_per_log", 100))
    log_path = Path(cfg_train["log"])
    ckpt_dir = Path(cfg_train["ckpt_dir"])

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    variables = create_variables(init_rng, model)
    params = variables["params"]
    stats = variables["stats"]
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, stats, opt_state,
                   node_in, edge_in, senders, receivers, target_delta_v, node_mask, edge_mask):

        def loss_fn(params):
            vars_ = {"params": params, "stats": stats}
            pred_delta_v = model.apply(
                vars_, node_in, edge_in, senders, receivers, edge_mask
            )
            err = pred_delta_v - target_delta_v
            mse = jnp.sum((err ** 2) * node_mask[:, None]) / (jnp.sum(node_mask) + 1e-8)
            return mse

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    def accumulate_stats(stats, node_in, edge_in, target_delta_v):
        vars_ = {"params": {}, "stats": stats}
        _, mutated = model.apply(
            vars_,
            node_in, edge_in, target_delta_v,
            method=MeshGraphNet.accumulate_norms,
            mutable=["stats"],
        )
        return mutated["stats"]

    total_num_nodes_seen = 0
    accumulate_stats_flag = True
    steps_per_traj = 1
    log_f = open(log_path, "w")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["epoch", "traj_id", "step", "loss", "elapsed_time"])

    epoch = 0
    traj_id = -1
    init_traj_it = lambda: threaded_trajectory_iterator(train_path, meta_path, traj_ids=train_traj_ids)
    traj_it = init_traj_it()
    last_log_time = time.perf_counter()
    for step in range(steps):
        if step % steps_per_traj == 0:
            traj = next(traj_it, None)
            if traj is None:
                log_f.flush()
                label = save_checkpoint(step, params, stats, epoch, ckpt_dir)
                print(f"Saving state @ step {step} | {ckpt_dir} | {label}")
                epoch += 1
                if epoch == 1 and total_num_nodes_seen < 1_000_000:
                    steps_per_traj = (1_000_000 - total_num_nodes_seen) // (traj_id + 1)
                else:
                    steps_per_traj = min((steps - step) // (traj_id + 1), 20)
                    accumulate_stats_flag = False
                traj_id = -1
                traj_it = init_traj_it()
                traj = next(traj_it)
            traj_id += 1

            # Load from traj dict to traj obj
            traj = Trajectory(traj, model.edge_feat_dim)
            N_pad = next_pow2(2*traj.N)
            E_pad = next_pow2(2*traj.E)
            graph = traj.get_graph()
            graph = jraph.pad_with_graphs(
                    graph,
                    n_node=N_pad,
                    n_edge=E_pad
                    )
            mask_padded = jnp.pad(
                    traj.mask,
                    (0, N_pad - traj.mask.shape[0]),
                    )
            edge_padding_mask = jraph.get_edge_padding_mask(graph) # shape: (E_pad,)

        # --- Train by sampling (t -> t+1) from the trajectory ---
        rng, node_in, target_delta_v = traj.get_random_data_node_in_out(rng)
        if accumulate_stats_flag:
            stats = accumulate_stats(stats, node_in, traj.edge_in, target_delta_v)
            total_num_nodes_seen += traj.N

        node_in_padded = jnp.pad(
                node_in,
                ((0, N_pad - node_in.shape[0]), (0,0)),
                )
        target_delta_v_padded = jnp.pad(
                target_delta_v,
                ((0, N_pad - target_delta_v.shape[0]), (0,0)),
                )

        # train params
        params, opt_state, loss = train_step(
            params, stats, opt_state,
            node_in_padded,
            graph.edges,
            graph.senders,
            graph.receivers,
            target_delta_v_padded,
            mask_padded,
            edge_padding_mask
        )

        if step % steps_per_log == 0:
            loss_val = jax.device_get(loss).item()

            now = time.perf_counter()
            elapsed_time = now - last_log_time
            last_log_time = now

            print(f"epoch {epoch:06d} | traj_id {traj_id:06d} | step {step:06d} | loss {loss_val:.6e} | elapsed_time {elapsed_time}")
            log_writer.writerow([epoch, traj_id, step, loss_val, elapsed_time])

    log_f.close()
    print(f"Logs saved at {str(log_path)}")

    label = save_checkpoint(step, params, stats, epoch, ckpt_dir, label="final")
    print(f"Final check point saved @ {ckpt_dir} | {label}")
    close_checkpointer()

