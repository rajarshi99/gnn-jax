import jax
import jax.numpy as jnp
from flax import linen as nn

class GNNLayer(nn.Module):
    msg: nn.Module
    node_update: nn.Module
    edge_update: nn.Module

    @nn.compact
    def __call__(self, h, e, senders, receivers, edge_mask=None, extra_node_params=None, extra_edge_params=None):
        hs = h[senders]
        hr = h[receivers]
        m = self.msg(e, hs, hr)

        if edge_mask is not None:
            m = m * edge_mask[:, None]
        agg = jax.ops.segment_sum(m, receivers, num_segments=h.shape[0])

        if extra_node_params is None:
            h_new = self.node_update(h, agg)
        else:
            h_new = self.node_update(h, agg, extra_node_params)

        if extra_edge_params is None:
            e_new = self.edge_update(e, m)
        else:
            e_new = self.edge_update(e, m, extra_edge_params)

        return h_new, e_new
