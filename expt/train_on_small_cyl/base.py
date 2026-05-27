"""
Trying to reproduce DeepMind MeshGraphNet CylinderFlow example.
Node input: velocity (vx,vy) + node_type one-hot.
Target: change to next-step velocity (delta vx, delta vy).

TF is used only to decode the TFRecord into numpy arrays via trajectory_iterator_np.
All graph construction (senders/receivers), edge features, and training are JAX.
"""
import argparse

import jax.numpy as jnp
from flax import linen as nn

from gnn_jax.data.cylinderflow_dm.load import NodeType
from gnn_jax.mlp import MLP
from gnn_jax.meshgraphnet import MeshGraphNet

from scripts.setup_run import setup_run
from scripts.run_model import run_model

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
        self.sow("intermediates", "messages", m)
        return m

class EdgeUpdate(nn.Module):
    def __call__(self, e, m):
        return e + m # Residual connection

# -------------------------
# Main
# -------------------------

def main():
    expt_name = "base"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval", "test", "check"], required=True)
    parser.add_argument("--config", type=str, help="Path to config.yaml in train mode")
    parser.add_argument("--run_dir", type=str, help="Path to dir for eval mode")
    parser.add_argument("--zeroE", action="store_true", help="Set input vel as zero ONLY in test mode")
    args = parser.parse_args()
    cfg = setup_run(args, expt_name)

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

    run_model(model, expt_name, args, cfg)

if __name__ == "__main__":
    main()

