from flax import linen as nn
from gnn_jax.mlp import MLP

from gnn_jax.data.cylinderflow_dm.load import NodeType
from gnn_jax.meshgraphnet import MeshGraphNet
from gnn_jax.dt_meshgraphnet import DtMeshGraphNet

import scripts.asymm as asymm
import scripts.tau as tau

# -------------------------
# Crucial MGN details
# -------------------------

import jax.numpy as jnp
from flax import linen as nn
from gnn_jax.mlp import MLP

class NodeUpdate(nn.Module):
    """
    Inputs
        -Node embeddings/feats
        -Aggregated messages
    Returns: Updated Node Embeddings with residual connection
    """
    latent_dim: int
    num_hidden_layers: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, h, agg):
        x = jnp.concatenate([h, agg], axis=-1)
        dh = MLP(
            [self.latent_dim]*self.num_hidden_layers + [h.shape[-1]],
            [nn.relu]*self.num_hidden_layers,
            use_bias = self.use_bias
        )(x)
        dh = nn.LayerNorm(use_bias=self.use_bias)(dh) # Layer Norm
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
        return nn.LayerNorm()(m)

class EdgeUpdate(nn.Module):
    def __call__(self, e, m):
        return e + m # Residual connection

def get_model(cfg, asymm_flag, tau_flag, expt_name):
    num_types = NodeType.SIZE
    node_feat_dim=num_types+2
    edge_feat_dim=3
    node_out_dim=2

    latent_dim = int(cfg["model"].get("latent_dim", 128))
    mp_steps = int(cfg["model"].get("message_passing_steps", 8))

    # MGN Defaults
    edge_update_factory=lambda l: EdgeUpdate()
    msg_compute_factory=lambda l: MessageCompute(
            latent_dim=latent_dim,
            num_hidden_layers=1,
            name=f"msg_{l}"
            )
    use_node_bias = True

    if asymm_flag:
        edge_update_factory = edge_update_factory=lambda l: asymm.EdgeUpdate()
        msg_compute_factory = lambda l: asymm.MessageCompute(
                latent_dim=latent_dim,
                num_hidden_layers=1,
                name=f"msg_{l}"
                )
        edge_feat_dim=1
        use_node_bias = False

    node_enc=MLP([latent_dim]*2, [nn.relu]*1, use_bias=use_node_bias, name="node_enc")
    edge_enc=MLP([latent_dim]*2, [nn.relu]*1, name="edge_enc")
    dec=MLP([latent_dim]*1 + [2], [nn.relu]*1, use_bias=use_node_bias, name="dec")
    node_update_factory=lambda l: NodeUpdate(
            latent_dim=latent_dim,
            num_hidden_layers=1,
            use_bias=use_node_bias,
            name=f"node_{l}"
            )

    if tau_flag:
        node_update_factory = lambda l: tau.NodeUpdate(
                latent_dim=latent_dim,
                num_hidden_layers=1,
                use_bias=use_node_bias,
                name=f"node_{l}"
                )

        model = DtMeshGraphNet(
                dtMax=float(cfg[expt_name].get("dt_max", 0.1)),
                latent_dim=latent_dim,
                node_feat_dim=node_feat_dim,
                node_enc=node_enc,
                edge_feat_dim=edge_feat_dim,
                edge_enc=edge_enc,
                message_passing_steps=mp_steps,
                node_update_factory=node_update_factory,
                edge_update_factory=edge_update_factory,
                msg_compute_factory=msg_compute_factory,
                node_out_dim=node_out_dim,
                dec=dec,
                use_node_bias=use_node_bias
                )

    else:
        model = MeshGraphNet(
                latent_dim=latent_dim,
                node_feat_dim=node_feat_dim,
                node_enc=node_enc,
                edge_feat_dim=edge_feat_dim,
                edge_enc=edge_enc,
                message_passing_steps=mp_steps,
                node_update_factory=node_update_factory,
                edge_update_factory=edge_update_factory,
                msg_compute_factory=msg_compute_factory,
                node_out_dim=node_out_dim,
                dec=dec,
                use_node_bias=use_node_bias
                )

    return model

