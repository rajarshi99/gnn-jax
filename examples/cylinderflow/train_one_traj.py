"""
Train on ONLY the first DeepMind CylinderFlow trajectory.
Node input: velocity (vx,vy) + node_type one-hot.
Target: next-step velocity (vx,vy).

TF is used only to decode the TFRecord into numpy arrays via trajectory_iterator_np.
All graph construction (senders/receivers), edge features, and training are JAX.
"""

import argparse
import yaml
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from gnn_jax.data.deepmind_cylinderflow import trajectory_iterator_np, NodeType
from gnn_jax.mlp import MLP
from gnn_jax.gnn_block import GNBlock, MeshGraphNet


def cells_to_bi_edges_jax(cells: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build unique bidirectional edges from triangle cells

    cells: shape (C, 3) int32
    returns senders, receivers: shape (E,) int32
    """
    a = cells[:, 0]
    b = cells[:, 1]
    c = cells[:, 2]

    e1 = jnp.stack([a, b], axis=1)
    e2 = jnp.stack([b, c], axis=1)
    e3 = jnp.stack([c, a], axis=1)
    edges = jnp.concatenate([e1, e2, e3], axis=0)      # (3C,2)

    rev = edges[:, ::-1]
    edges = jnp.concatenate([edges, rev], axis=0)      # (6C,2)

    # unique undirected+directed edges
    edges = jnp.unique(edges, axis=0)                  # (E,2)

    senders = edges[:, 0].astype(jnp.int32)
    receivers = edges[:, 1].astype(jnp.int32)
    return senders, receivers


# -------------------------
# Training utilities
# -------------------------

def create_state(rng, model, lr):
    # dummy init (only feature dims matter)
    N = 4
    E = 6
    node_dim = 2 + model.node_type_dim
    node_in = jnp.zeros((N, node_dim), dtype=jnp.float32)
    edge_in = jnp.zeros((E, model.edge_feat_dim), dtype=jnp.float32)
    senders = jnp.zeros((E,), dtype=jnp.int32)
    receivers = jnp.zeros((E,), dtype=jnp.int32)

    params = model.init(rng, node_in, edge_in, senders, receivers)["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, node_in, edge_in, senders, receivers, target_v, mask):
    def loss_fn(params):
        delta_v = state.apply_fn({"params": params}, node_in, edge_in, senders, receivers)
        pred_v = node_in[:, :2] + delta_v
        err = pred_v - target_v
        mse = jnp.sum((err ** 2) * mask[:, None]) / (jnp.sum(mask) + 1e-8)
        return mse

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


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
    steps = int(cfg["train"].get("steps", 5000))
    seed = int(cfg["train"].get("seed", 0))
    num_types = NodeType.SIZE

    # --- Load only the first trajectory (TF decode -> numpy) ---
    traj = next(trajectory_iterator_np(train_path, meta_path))

    # Convert required arrays to JAX once
    mesh_pos = jnp.asarray(traj["mesh_pos"][0], dtype=jnp.float32)       # (N,2)
    node_type = jnp.asarray(traj["node_type"][0, :, 0], dtype=jnp.int32) # (N,)
    cells = jnp.asarray(traj["cells"][0], dtype=jnp.int32)               # (C,3)
    vel = jnp.asarray(traj["velocity"], dtype=jnp.float32)               # (T,N,2)

    N = mesh_pos.shape[0]
    T = vel.shape[0]

    # --- Build senders/receivers ---
    senders, receivers = cells_to_bi_edges_jax(cells)

    # --- Edge features: dx, dy, dist ---
    rel = mesh_pos[receivers] - mesh_pos[senders]                         # (E,2)
    dist = jnp.linalg.norm(rel, axis=1, keepdims=True)                    # (E,1)
    edge_in = jnp.concatenate([rel, dist], axis=1).astype(jnp.float32)    # (E,3)

    # --- Node type one-hot ---
    node_type_oh = jax.nn.one_hot(node_type, num_classes=num_types, dtype=jnp.float32)  # (N,num_types)

    # --- Loss mask: only NORMAL and OUTFLOW nodes ---
    mask = (
            (node_type == NodeType.NORMAL) |
            (node_type == NodeType.OUTFLOW)
            ).astype(jnp.float32)        # (N,)

    # --- Model/state ---
    model = MeshGraphNet(latent_dim=latent_dim, message_passing_steps=mp_steps, node_type_dim=num_types)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    state = create_state(init_rng, model, lr)

    print(f"First trajectory loaded: N={int(N)}, E={int(senders.shape[0])}, T={int(T)}")

    # --- Train by sampling (t -> t+1) from the ONE trajectory ---
    for step in range(steps):
        rng, sub = jax.random.split(rng)
        t = int(jax.random.randint(sub, (), 0, T - 1))

        v_t = vel[t]       # (N,2)
        v_t1 = vel[t + 1]  # (N,2)

        # node input = [vx, vy, node_type_onehot]
        node_in = jnp.concatenate([v_t, node_type_oh], axis=-1)           # (N,2+T)

        state, loss = train_step(state, node_in, edge_in, senders, receivers, v_t1, mask)

        if step % 100 == 0:
            print(f"step {step:06d} | loss {float(loss):.6e}")

    print("Done (first trajectory only).")

if __name__ == "__main__":
    main()
