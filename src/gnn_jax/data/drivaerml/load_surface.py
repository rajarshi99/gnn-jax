import trimesh
import pyvista as pv

import jax
import jax.numpy as jnp
import jraph

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import json


def get_io(stl_path, vtp_path):

    # Use mesh to make graph
    mesh = trimesh.load(stl_path)
    # Handle Scene
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )
    vertices = mesh.vertices
    mesh.fix_normals()

    node_feats = np.concatenate(
            [vertices, mesh.vertex_normals],
            axis = -1
            )

    faces = mesh.faces
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    senders = np.concatenate([i, j, k, j, k, i])
    receivers = np.concatenate([j, k, i, i, j, k])

    rel = vertices[receivers] - vertices[senders]
    dist = np.linalg.norm(rel, axis=-1, keepdims=True)
    edge_feats = np.concatenate([rel, dist], axis=-1)

    graph = jraph.GraphsTuple(
        nodes=jnp.array(node_feats),
        edges=jnp.array(edge_feats),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array([node_feats.shape[0]]),
        n_edge=jnp.array([edge_feats.shape[0]]),
        globals=None,
    )

    # Use data to make target
    data = pv.read(vtp_path)
    data = data.cell_data_to_point_data()

    tree = cKDTree(data.points)
    dist, idx = tree.query(vertices, k=5)
    weights = 1.0 / (dist + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    Cp = data.point_data["CpMeanTrim"]
    Cp_out = np.sum(weights * Cp[idx], axis=1, keepdims=True)

    shear_stress = data.point_data["wallShearStressMeanTrim"]
    shear_stress_out = np.sum(weights[...,None] * shear_stress[idx], axis=1, keepdims=True)

    target_node_out = np.concatenate(
            [Cp_out[:,None], shear_stress_out],
            axis=-1
            )

    return graph, target_node_out

class DrivAerIterator:
    def __init__(self, rng, base_path, stats_path, mode="train"):
        self.base_path = Path(base_path)
        stats_path = Path(stats_path)

        run_dirs = np.array([p for p in self.base_path.iterdir() if p.is_dir()])

        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            split = stats["split"]
            self.max_nodes = stats["max_nodes"]
            self.max_edges = stats["max_edges"]

        else:
            print(f"Creating dataset split as {stats_path} not found")
            perm = jax.random.permutation(rng, len(run_dirs))
            shuffled = run_dirs[perm]

            N = len(shuffled)
            N_train = int(0.8*N)
            N_val = int(0.1*N)

            train_dirs = list(shuffled[:N_train])
            val_dirs = list(shuffled[N_train:N_train+N_val])
            test_dirs = list(shuffled[N_train+N_val:])

            split = {
                    "train": train_dirs,
                    "val": val_dirs,
                    "test": test_dirs
                    }
            print(split)

            self.max_nodes = 0
            self.max_edges = 0
            for run_dir in train_dirs:
                run_path = self.base_path / run_dir
                stl_path = list(run_path.glob("drivaer_*_single_solid.stl"))[0]
                vtp_path = list(run_path.glob("boundary_*.vtp"))[0]
                graph, target_node_out = get_io(stl_path, vtp_path)
                if self.max_nodes < graph.n_node[0]:
                    self.max_nodes = graph.n_node[0]
                if self.max_edges < graph.n_edge[0]:
                    self.max_nodes = graph.n_edge[0]
                print(f"{run_dir} | num_nodes {graph.n_node[0]} | num_edges {graph.n_edge[0]}")
            
            stats = {
                    "split": split,
                    "max_nodes": self.max_nodes,
                    "max_edges": self.max_edges,
                    }

            with open(stats_path, "w") as f:
                json.dump(stats, f)

        self.run_dirs = split[mode]

    def __len__(self):
        return len(self.run_dirs)

    def __iter__(self):
        for run_dir in self.run_dirs:
            print(run_dir)
            run_path = self.base_path / run_dir
            stl_path = list(run_path.glob("drivaer_*_single_solid.stl"))[0]
            vtp_path = list(run_path.glob("boundary_*.vtp"))[0]
            graph, target_node_out = get_io(stl_path, vtp_path)

            # Pad the graph = bla bla

            yield graph, target_node_out

