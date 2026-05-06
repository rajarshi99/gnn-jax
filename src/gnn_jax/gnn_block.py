import jax
import jax.numpy as jnp
from flax import linen as nn

from gnn_jax.mlp import MLP

class GNBlock(nn.Module):
    msg_mlp: nn.Module
    upd_mlp: nn.Module

    @nn.compact
    def __call__(self, h, e, senders, receivers):
        hs = h[senders]
        hr = h[receivers]
        m_in = jnp.concatenate([e, hs, hr], axis=-1)
        m = self.msg_mlp(m_in)  # (E, latent)

        agg = jax.ops.segment_sum(m, receivers, num_segments=h.shape[0])  # (N, latent)
        u_in = jnp.concatenate([h, agg], axis=-1)
        dh = self.upd_mlp(u_in)
        return h + dh

class MeshGraphNet(nn.Module):
    latent_dim: int = 128
    message_passing_steps: int = 8
    node_type_dim: int = 7   # NORMAL..WALL_BOUNDARY
    edge_feat_dim: int = 3   # dx, dy, dist

    @nn.compact
    def __call__(self, node_in, edge_in, senders, receivers):
        node_enc = MLP((self.latent_dim, self.latent_dim), (nn.relu, nn.relu), name="node_enc")
        edge_enc = MLP((self.latent_dim, self.latent_dim), (nn.relu, nn.relu), name="edge_enc")

        h = node_enc(node_in)
        e = edge_enc(edge_in)

        for k in range(self.message_passing_steps):
            msg_mlp = MLP((self.latent_dim, self.latent_dim), (nn.relu, nn.relu), name=f"msg_{k}")
            upd_mlp = MLP((self.latent_dim, self.latent_dim), (nn.relu, nn.relu), name=f"upd_{k}")
            h = GNBlock(msg_mlp=msg_mlp, upd_mlp=upd_mlp, name=f"gn_{k}")(h, e, senders, receivers)

        # decoder outputs delta_v (dvx, dvy)
        dec = MLP((self.latent_dim, 2), (nn.relu, (lambda x: x)), name="dec")
        delta_v = dec(h)
        return delta_v


