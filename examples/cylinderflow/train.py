"""
Trying to reproduce DeepMind MeshGraphNet CylinderFlow example.
Node input: velocity (vx,vy) + node_type one-hot.
Target: change to next-step velocity (delta vx, delta vy).

TF is used only to decode the TFRecord into numpy arrays via trajectory_iterator_np.
All graph construction (senders/receivers), edge features, and training are JAX.
"""

import argparse
import yaml
from pathlib import Path
import csv
import time

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn

from gnn_jax.data.deepmind_cylinderflow import threaded_trajectory_iterator, NodeType
from gnn_jax.mlp import MLP
from gnn_jax.meshgraphnet import MeshGraphNet, save_checkpoint

import jraph

# -------------------------
# Crucial MGN details
# -------------------------
class NodeUpdate(nn.Module):
    """
    Inputs
        -Node embeddings/feats
        -Aggregated messages
    Returns: Updated Node Embeddings with residual connection
    """
    latent_dim: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, h, agg):
        x = jnp.concatenate([h, agg], axis=-1)
        dh = MLP(
            [self.latent_dim]*self.num_hidden_layers + [h.shape[-1]],
            [nn.relu]*self.num_hidden_layers
        )(x)
        return h + dh # Residual connection

class EdgeUpdate(nn.Module):
    def __call__(self, e, m):
        return e + m # Residual connection

# -------------------------
# Training utilities
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

    variables = model.init(rng, node_in, edge_in, senders, receivers)
    return variables["params"], variables["stats"]

def get_random_data_node_in_out(rng, vel, node_type_oh):
    rng, sub = jax.random.split(rng)
    T = vel.shape[0]
    t = int(jax.random.randint(sub, (), 0, T - 1))
    v_t = (vel[t])          # (N,2)
    v_t1 = (vel[t+1])       # (N,2)
    target_delta_v = v_t1 - v_t
    node_in = jnp.concatenate([v_t, node_type_oh], axis=-1)
    return rng, node_in, target_delta_v

def get_data_edge_in(mesh_pos, senders, receivers):
    rel = mesh_pos[receivers] - mesh_pos[senders]                         # (E,2)
    dist = jnp.linalg.norm(rel, axis=1, keepdims=True)                    # (E,1)
    edge_in = jnp.concatenate([rel, dist], axis=1).astype(jnp.float32)    # (E,3)
    return edge_in

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["dataset"]["dir"])
    train_path = data_dir / cfg["dataset"]["train"]
    meta_path = data_dir / cfg["dataset"]["meta"]

    latent_dim = int(cfg["model"].get("latent_dim", 128))
    mp_steps = int(cfg["model"].get("message_passing_steps", 8))
    lr = float(cfg["train"].get("learning_rate", 1e-4))
    steps = int(cfg["train"].get("steps", 500))
    epochs = int(cfg["train"].get("epochs", 5))
    max_steps_per_traj_for_stat_acc = int(cfg["train"].get("max_steps_per_traj_for_stat_acc", 5))
    seed = int(cfg["train"].get("seed", 0))
    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    log_path = Path(cfg["output"]["log"])
    num_types = NodeType.SIZE

    # --- Model/state ---
    model = MeshGraphNet(
            latent_dim=latent_dim,
            node_feat_dim=num_types+2,
            node_enc=MLP([latent_dim]*2, [nn.relu]*1, name="node_enc"),
            edge_feat_dim=3,
            edge_enc=MLP([latent_dim]*2, [nn.relu]*1, name="edge_enc"),
            message_passing_steps=mp_steps,
            node_update_factory=lambda l: NodeUpdate(
                latent_dim=latent_dim,
                num_hidden_layers=1,
                name=f"node_{l}"
                ),
            edge_update_factory=lambda l: EdgeUpdate(),
            msg_compute_factory=lambda l: MLP(
                [latent_dim]*2,
                [nn.relu]*1,
                name=f"msg_{l}"),
            node_out_dim=2,
            dec=MLP([latent_dim]*1 + [2], [nn.relu]*1, name="dec"),
            )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params, stats = create_variables(init_rng, model)
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

    traj_it = threaded_trajectory_iterator(train_path, meta_path)

    epoch = 0
    traj_id = 0
    total_N = 0
    max_steps_per_traj_for_stat_acc = 1

    log_f = open(log_path, "w")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["epoch", "traj_id", "step", "loss", "elapsed_time"])
    last_log_time = time.perf_counter()

    while epoch < epochs:
        traj = next(traj_it, None)
        if traj is None:
            log_f.flush()
            save_checkpoint(step, params, stats, epoch, ckpt_dir)
            print(f"Saving end of epoch {epoch} in {ckpt_dir}")

            if epoch == 0:
                max_steps_per_traj_for_stat_acc = int(0.5 + (2_000_000 - total_N) / traj_id)
                print(f"max_steps_per_traj_for_stat_acc {max_steps_per_traj_for_stat_acc}")
                total_N = 0

            epoch += 1
            traj_id = 0
            traj_it = threaded_trajectory_iterator(train_path, meta_path)
            continue

        # Convert required arrays to JAX
        beg_load2jax = time.perf_counter()
        mesh_pos = jnp.asarray(traj["mesh_pos"][0], dtype=jnp.float32)       # (N,2)
        node_type = jnp.asarray(traj["node_type"][0, :, 0], dtype=jnp.int32) # (N,)
        vel = jnp.asarray(traj["velocity"], dtype=jnp.float32)               # (T,N,2)
        senders = jnp.asarray(traj["senders"], dtype=jnp.int32)              # (E,)
        receivers = jnp.asarray(traj["receivers"], dtype=jnp.int32)          # (E,)
        end_load2jax = time.perf_counter()
        print(f"Time to load into JAX {end_load2jax - beg_load2jax}")

        N = mesh_pos.shape[0]
        # T = vel.shape[0]

        # --- Edge features: dx, dy, dist ---
        edge_in = get_data_edge_in(mesh_pos, senders, receivers)

        # --- Node type one-hot ---
        node_type_oh = jax.nn.one_hot(node_type, num_classes=num_types, dtype=jnp.float32)  # (N,num_types)

        # --- Loss mask: only NORMAL and OUTFLOW nodes ---
        mask = (
                (node_type == NodeType.NORMAL) |
                (node_type == NodeType.OUTFLOW)
                ).astype(jnp.float32)        # (N,)

        # --- Data: Random time step from velocity ---
        rng, node_in, target_delta_v = get_random_data_node_in_out(rng, vel, node_type_oh)
        graph = jraph.GraphsTuple(
            nodes=node_in,          # (N, F)
            edges=edge_in,          # (E, D)
            senders=senders,        # (E,)
            receivers=receivers,    # (E,)
            n_node=jnp.array([node_in.shape[0]]),
            n_edge=jnp.array([edge_in.shape[0]]),
            globals=None,
        )
        graph = jraph.pad_with_graphs(
                graph,
                n_node=next_pow2(node_in.shape[0]),
                n_edge=next_pow2(edge_in.shape[0])
                )
        mask_padded = jnp.pad(
                mask,
                (0, graph.nodes.shape[0] - mask.shape[0]),
                )
        node_padding_mask = jraph.get_node_padding_mask(graph) # shape: (N_pad,)
        edge_padding_mask = jraph.get_edge_padding_mask(graph) # shape: (N_pad,)
        train_node_mask = node_padding_mask * mask_padded

        # --- Train by sampling (t -> t+1) from the trajectory ---
        for step in range(steps):
            rng, node_in, target_delta_v = get_random_data_node_in_out(rng, vel, node_type_oh)
            target_delta_v_padded = jnp.pad(
                    target_delta_v,
                    ((0, graph.nodes.shape[0] - target_delta_v.shape[0]), (0,0)),
                    )
            node_in_padded = jnp.pad(
                    node_in,
                    ((0, graph.nodes.shape[0] - node_in.shape[0]), (0,0)),
                    )

            if epoch <= 1 and step < max_steps_per_traj_for_stat_acc:
                stats = accumulate_stats(stats, node_in, edge_in, target_delta_v)
                total_N += N

            # train params
            params, opt_state, loss = train_step(
                params, stats, opt_state,
                node_in_padded,
                graph.edges,
                graph.senders,
                graph.receivers,
                target_delta_v_padded,
                train_node_mask,
                edge_padding_mask
            )

            if step % 100 == 0:
                loss_val = float(loss)

                now = time.perf_counter()
                elapsed_time = now - last_log_time
                last_log_time = now

                print(f"epoch {epoch:06d} | traj_id {traj_id:06d} | step {step:06d} | loss {loss_val:.6e} | elapsed_time {elapsed_time}")
                log_writer.writerow([epoch, traj_id, step, loss_val, elapsed_time])

        traj_id += 1

    log_f.close()
    print("Logs saved at {str(log_path)}")

if __name__ == "__main__":
    main()

