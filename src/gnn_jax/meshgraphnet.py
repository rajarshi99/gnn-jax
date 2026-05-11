from flax import linen as nn

from gnn_jax.gnn_layer import GNNLayer
from gnn_jax.normalizer import Normalizer

from pathlib import Path
import orbax.checkpoint as ocp

class MeshGraphNet(nn.Module):
    latent_dim: int

    node_feat_dim: int
    node_enc: nn.Module

    edge_feat_dim: int
    edge_enc: nn.Module

    message_passing_steps: int
    node_update_factory: callable
    edge_update_factory: callable
    msg_compute_factory: callable

    node_out_dim: int
    dec: nn.Module

    def setup(self):
        self.node_norm = Normalizer(feature_dim=self.node_feat_dim, name="node_norm")
        self.edge_norm = Normalizer(feature_dim=self.edge_feat_dim, name="edge_norm")

        self.gnn_layers = [
                GNNLayer(
                    msg = self.msg_compute_factory(l),
                    node_update = self.node_update_factory(l),
                    edge_update = self.edge_update_factory(l),
                    )
                for l in range(self.message_passing_steps)
                ]

        self.out_data_norm = Normalizer(feature_dim=self.node_out_dim, name="out_data_norm")

    def __call__(self, node_in, edge_in, senders, receivers):
        h = self.node_enc(self.node_norm.normalize(node_in))
        e = self.edge_enc(self.edge_norm.normalize(edge_in))

        for gnn_layer in self.gnn_layers:
            h, e = gnn_layer(h, e, senders, receivers)

        # decoder outputs delta_v (dvx, dvy)
        delta_v = self.dec(h)
        return self.out_data_norm.denormalize(delta_v)

    def accumulate_norms(self, node_in, edge_in, delta_v_gt):
        self.node_norm.accumulate(node_in)
        self.edge_norm.accumulate(edge_in)
        self.out_data_norm.accumulate(delta_v_gt)


# -------------------------------------------------------------
# Checkpointing helpers
# -------------------------------------------------------------

_checkpointer = ocp.StandardCheckpointer()

def save_checkpoint(step, params, stats, epoch, ckpt_dir: Path):
    state = {
        "params": params,
        "stats": stats,
        "step": step,
        "epoch": epoch,
    }
    _checkpointer.save(ckpt_dir / f"model_{epoch:06d}", state)

def load_checkpoint(ckpt_path: Path):
    return _checkpointer.restore(ckpt_path)
