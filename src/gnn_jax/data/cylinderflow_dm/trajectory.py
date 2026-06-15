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

    def get_random_data_in_out(self, rng, max_tstep, add_noise=True):
        # First decide n_tstep
        if max_tstep is None:
            n_tstep = 1
        else:
            rng, sub = jax.random.split(rng)
            n_tstep = int(jax.random.randint(sub, (), 1, max_tstep+1))

        # Then decide current time t1 and next time step t2
        rng, sub = jax.random.split(rng)
        t1 = int(jax.random.randint(sub, (), 0, self.T - n_tstep))
        t2 = t1 + n_tstep
        v_t1 = (self.vel[t1])       # (N,2)
        v_t2 = (self.vel[t2])       # (N,2)

        if add_noise:
            rng, noise_rng = jax.random.split(rng)
            noise_std = 2e-2 # from MGN paper Appendix
            v_t1 += noise_std * jax.random.normal(noise_rng, v_t1.shape)

        target_delta_v = v_t2 - v_t1
        node_in = jnp.concatenate([v_t1, self.node_type_oh], axis=-1)
        return rng, n_tstep, node_in, target_delta_v

