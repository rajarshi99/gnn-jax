import trimesh
import pyvista as pv

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import json


def get_io(stl_path, vtp_path):
    io_dict = {}

    # Use mesh to make graph
    mesh = trimesh.load(stl_path)
    # Handle Scene
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )
    vertices = mesh.vertices
    mesh.fix_normals()

    io_dict["node_in"] = {
                "vertices": vertices,
                "normals" : mesh.vertex_normals
                }

    faces = mesh.faces
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    senders = np.concatenate([i, j, k, j, k, i])
    receivers = np.concatenate([j, k, i, i, j, k])

    rel_pos = vertices[receivers] - vertices[senders]
    distance = np.linalg.norm(rel, axis=-1, keepdims=True)
    io_dict["edge_in"] = {
            "rel_pos" : rel_pos,
            "distance": distance
            }

    # Put somewhere else
    # graph = jraph.GraphsTuple(
    #     nodes=jnp.array(node_feats),
    #     edges=jnp.array(edge_feats),
    #     senders=jnp.array(senders),
    #     receivers=jnp.array(receivers),
    #     n_node=jnp.array([node_feats.shape[0]]),
    #     n_edge=jnp.array([edge_feats.shape[0]]),
    #     globals=None,
    # )

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

    io_dict["node_out"] = {
            "Cp": Cp_out,
            "shear_stree": shear_stress_out
            }

    return io_dict

class AccumulateStats:
    def __init__(self):
        self.stats = {}

    def push_sample(self, sample):
        for k,v in sample.items():
            if k in self.stats:
                self.stats[k]["cnt"].append(v.shape[0])
                self.stats[k]["avg"].append(np.mean(v, axis=0))
                self.stats[k]["var"].append(np.var(v, axis=0))
            else:
                self.stats[k] = {
                        "cnt" : [v.shape[0]],
                        "avg" : [np.mean(v, axis=0)],
                        "var" : [np.var(v, axis=0)]
                        }
        return v.shape[0]

    def get_avg_std(self):
        result = {}
        for name,stats in self.stats.items():
            cnt = np.array(stats["cnt"])
            avg = np.array(stats["avg"])
            var = np.array(stats["var"])

            glob_avg = np.sum(avg*cnt[:,None], axis=0) / np.sum(cnt)
            glob_var = np.sum(cnt[:,None] * (var + (avg - glob_avg)**2), axis=0) / np.sum(cnt)
            result[name] = {
                    "avg": glob_avg,
                    "std": np.sqrt(glob_var)
                    }

        return result



class DrivAerIterator:
    def __init__(self, rng, cfg_data, mode="train"):
        self.base_path = Path(cfg_data["dir"])
        stats_path = Path(cfg_data["stats"])

        run_dirs = np.array([p for p in self.base_path.iterdir() if p.is_dir()])

        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            split = stats["split"]
            self.max_nodes = stats["max_nodes"]
            self.max_edges = stats["max_edges"]
            self.node_in_stats = stats["node_in"]
            self.edge_in_stats = stats["edge_in"]
            self.node_out_stats = stats["node_out"]

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

            bad_id = []
            good_id = []
            self.max_nodes = 0
            self.max_edges = 0
            node_in = AccumulateStats()
            edge_in = AccumulateStats()
            node_out = AccumulateStats()

            for i,run_dir in enumerate(train_dirs):
                run_path = self.base_path / run_dir
                stl_path = list(run_path.glob("drivaer_*_single_solid.stl"))[0]
                vtp_path = list(run_path.glob("boundary_*.vtp"))[0]
                try:
                    io_dict = get_io(stl_path, vtp_path)
                    n_node = node_in.push_sample(io_dict["node_in"])
                    n_edge = edge_in.push_sample(io_dict["edge_in"])
                    n_node = node_out.push_sample(io_dict["node_out"])

                    if self.max_nodes < n_node:
                        self.max_nodes = n_node
                    if self.max_edges < n_edge:
                        self.max_nodes = n_edge

                    print(f"Train ID: {i}")
                    good_id.append(i)
                except:
                    print(f"BAD run_dir {run_dir}")
                    bad_id.append(i)
                    continue
            
            train_dirs = [train_dirs[i] for i in good_id]
            bad_dirs = [train_dirs[i] for i in bad_id]
            split = {
                    "train" : train_dirs,
                    "val"   : val_dirs,
                    "test"  : test_dirs,
                    "bad"   : bad_dirs
                    }
            print(split)

            self.node_in_stats = node_in.get_avg_std()
            self.edge_in_stats = edge_in.get_avg_std()
            self.node_out_stats = node_out.get_avg_std()

            stats = {
                    "split"    : split,
                    "max_nodes": self.max_nodes,
                    "max_edges": self.max_edges,
                    "node_in"  : self.node_in_stats,
                    "edge_in"  : self.edge_in_stats,
                    "node_out" : self.node_out_stats
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

