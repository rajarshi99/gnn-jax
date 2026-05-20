import jax
import jax.numpy as jnp
import jraph

from gnn_jax.data.cylinderflow_dm.load import NodeType

class Trajectory:
    def __init__(self, traj, edge_feat_dim=3):
        # Convert required arrays to JAX
        mesh_pos = jnp.asarray(traj["mesh_pos"][0], dtype=jnp.float32)            # (N,2)
        node_type = jnp.asarray(traj["node_type"][0, :, 0], dtype=jnp.int32)      # (N,)
        self.vel = jnp.asarray(traj["velocity"], dtype=jnp.float32)                    # (T,N,2)
        self.senders = jnp.asarray(traj["senders"], dtype=jnp.int32)                   # (E,)
        self.receivers = jnp.asarray(traj["receivers"], dtype=jnp.int32)               # (E,)

        self.N = mesh_pos.shape[0]
        self.T = self.vel.shape[0]
        self.E = self.senders.shape[0]

        # --- Edge features: (dx, dy, dist) or (dist) ---
        rel = mesh_pos[self.receivers] - mesh_pos[self.senders]              # (E,2)
        dist = jnp.linalg.norm(rel, axis=1, keepdims=True)                             # (E,1)
        if edge_feat_dim == 1:
            self.edge_in = dist
        elif edge_feat_dim == 2:
            self.edge_in = rel
        else:
            self.edge_in = jnp.concatenate([rel, dist], axis=1).astype(jnp.float32)    # (E,3)

        # --- Node type one-hot ---
        self.node_type_oh = jax.nn.one_hot(node_type, num_classes=NodeType.SIZE, dtype=jnp.float32)  # (N,num_types)

        # --- Loss mask: only NORMAL and OUTFLOW nodes ---
        self.mask = (
                (node_type == NodeType.NORMAL) |
                (node_type == NodeType.OUTFLOW)
                ).astype(jnp.float32)        # (N,)

    def get_graph(self):
        node_in = jnp.concatenate([self.vel[0], self.node_type_oh], axis=-1)
        return jraph.GraphsTuple(
            nodes=node_in,               # (N, F)
            edges=self.edge_in,          # (E, D)
            senders=self.senders,        # (E,)
            receivers=self.receivers,    # (E,)
            n_node=jnp.array([node_in.shape[0]]),
            n_edge=jnp.array([self.edge_in.shape[0]]),
            globals=None,
        )

    def get_random_data_node_in_out(self, rng):
        rng, sub = jax.random.split(rng)
        t = int(jax.random.randint(sub, (), 0, self.T - 1))
        v_t = (self.vel[t])          # (N,2)
        v_t1 = (self.vel[t+1])       # (N,2)
        target_delta_v = v_t1 - v_t
        node_in = jnp.concatenate([v_t, self.node_type_oh], axis=-1)
        return rng, node_in, target_delta_v

