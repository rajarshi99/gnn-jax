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
import json

import jax.numpy as jnp

from flax import linen as nn

from gnn_jax.data.deepmind_cylinderflow import NodeType
from gnn_jax.mlp import MLP
from gnn_jax.meshgraphnet import MeshGraphNet, save_checkpoint, close_checkpointer

from scripts.train import train

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

class MessageCompute(nn.Module):
    """
    Inputs
        -Edge embeddings/feats
        -Sender node embeddings
        -Receiver node embeddings
    Returns: Edge message same dimension as e
    """
    latent_dim: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, e, hs, hr):
        m_in = jnp.concatenate([e, hs, hr], axis=-1)
        m = MLP(
            [self.latent_dim]*self.num_hidden_layers + [e.shape[-1]],
            [nn.relu]*self.num_hidden_layers
        )(m_in)
        return m

class EdgeUpdate(nn.Module):
    def __call__(self, e, m):
        return e + m # Residual connection

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["dataset"]["dir"])
    train_path = data_dir / cfg["dataset"]["train"]
    meta_path = data_dir / cfg["dataset"]["meta"]

    latent_dim = int(cfg["model"].get("latent_dim", 128))
    mp_steps = int(cfg["model"].get("message_passing_steps", 8))
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
            msg_compute_factory=lambda l: MessageCompute(
                latent_dim=latent_dim,
                num_hidden_layers=1,
                name=f"msg_{l}"),
            node_out_dim=2,
            dec=MLP([latent_dim]*1 + [2], [nn.relu]*1, name="dec"),
            )

    split_path = Path(cfg["custom"]["split_path"])
    with open(split_path, "r") as f:
        split = json.load(f)

    if args.mode == "train":
        train_traj_ids = split["train_traj_ids"]
        train(model, cfg["train_base"], train_path, meta_path, train_traj_ids)

if __name__ == "__main__":
    main()

