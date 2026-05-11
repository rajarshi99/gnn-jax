import jax
import jax.numpy as jnp
from flax import linen as nn

class GNNLayer(nn.Module):
    msg: nn.Module
    node_update: nn.Module
    edge_update: nn.Module

    @nn.compact
    def __call__(self, h, e, senders, receivers):
        hs = h[senders]
        hr = h[receivers]
        m_in = jnp.concatenate([e, hs, hr], axis=-1)
        m = self.msg(m_in)

        agg = jax.ops.segment_sum(m, receivers, num_segments=h.shape[0])
        # u_in = jnp.concatenate([h, agg], axis=-1)
        # dh = self.upd_mlp(u_in)

        h_new = self.node_update(h, agg)
        e_new = self.edge_update(e, m)
        return h_new, e_new
