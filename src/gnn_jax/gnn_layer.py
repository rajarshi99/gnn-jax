import jax
import jax.numpy as jnp
from flax import linen as nn

class GNNLayer(nn.Module):
    msg: nn.Module
    node_update: nn.Module
    edge_update: nn.Module

    @nn.compact
    def __call__(self, h, e, senders, receivers, edge_mask=None):
        hs = h[senders]
        hr = h[receivers]
        m_in = jnp.concatenate([e, hs, hr], axis=-1)
        m = self.msg(m_in)

        if edge_mask is not None:
            m = m * edge_mask[:, None]
        agg = jax.ops.segment_sum(m, receivers, num_segments=h.shape[0])

        h_new = self.node_update(h, agg)
        e_new = self.edge_update(e, m)
        return h_new, e_new
