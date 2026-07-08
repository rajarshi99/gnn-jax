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
    io_dict["senders"] = senders
    io_dict["receivers"] = receivers

    rel_pos = vertices[receivers] - vertices[senders]
    distance = np.linalg.norm(rel_pos, axis=-1, keepdims=True)
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
            "shear_stress": shear_stress_out
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
                    "avg": glob_avg.tolist(),
                    "std": np.sqrt(glob_var).tolist()
                    }

        return result

class DrivAerIterator:
    def __init__(self, cfg_data, mode="train"):
        self.base_path = Path(cfg_data["dir"])
        self.stats_path = Path(cfg_data["stats"])
        self.mode = mode

        run_dirs = np.array([p.name for p in self.base_path.iterdir() if p.is_dir()])

        if self.stats_path.exists():
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
            self.split = stats["split"]
            self.max_nodes = stats["max_nodes"]
            self.max_edges = stats["max_edges"]
            self.node_in_stats = stats["node_in"]
            self.edge_in_stats = stats["edge_in"]
            self.node_out_stats = stats["node_out"]

        else:
            print(f"Creating dataset split as {self.stats_path} not found")
            perm = np.random.permutation(len(run_dirs))
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
                print(run_dir, type(run_dir))
                run_path = self.base_path / run_dir
                stl_path = list(run_path.glob("drivaer_*.stl"))[0]
                vtp_path = list(run_path.glob("boundary_*.vtp"))[0]
                try:
                    io_dict = get_io(stl_path, vtp_path)
                    n_node = node_in.push_sample(io_dict["node_in"])
                    n_edge = edge_in.push_sample(io_dict["edge_in"])
                    n_node = node_out.push_sample(io_dict["node_out"])

                    if self.max_nodes < n_node:
                        self.max_nodes = n_node
                    if self.max_edges < n_edge:
                        self.max_edges = n_edge

                    print(f"Train ID: {i}")
                    good_id.append(i)
                except:
                    print(f"BAD run_dir {run_dir}")
                    bad_id.append(i)

                if i == 5:
                    break
            
            bad_dirs = [train_dirs[i] for i in bad_id]
            new_train_dirs = [train_dirs[i] for i in good_id]
            self.split = {
                    "train" : new_train_dirs,
                    "val"   : val_dirs,
                    "test"  : test_dirs,
                    "bad"   : bad_dirs
                    }
            print(self.split)

            self.node_in_stats = node_in.get_avg_std()
            self.edge_in_stats = edge_in.get_avg_std()
            self.node_out_stats = node_out.get_avg_std()

            self.save_stats()

    def save_stats(self):
        stats = {
                "split"    : self.split,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
                "node_in"  : self.node_in_stats,
                "edge_in"  : self.edge_in_stats,
                "node_out" : self.node_out_stats
                }

        with open(self.stats_path, "w") as f:
            json.dump(stats, f)

    def __len__(self):
        return len(self.split[self.mode])

    def __iter__(self):
        bad_id = []
        good_id = []
        for i,run_dir in enumerate(self.split[self.mode]):
            print(run_dir)
            run_path = self.base_path / run_dir
            stl_path = list(run_path.glob("drivaer_*.stl"))[0]
            vtp_path = list(run_path.glob("boundary_*.vtp"))[0]

            try:
                io_dict = get_io(stl_path, vtp_path)
                good_id.append(i)
                yield io_dict
            except:
                print(f"BAD run_dir {run_dir}")
                bad_id.append(i)

        for i in bad_id:
            self.split["bad"].append(self.split[self.mode][i])
        new_run_dir_list = [self.split[self.mode][i] for i in good_id]
        self.split[self.mode] = new_run_dir_list
        self.save_stats()


