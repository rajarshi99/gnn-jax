import json
import tensorflow as tf
import numpy as np
import enum

import threading
import queue

class NodeType(enum.IntEnum):
    """
    Integer codes identifying the physical role of mesh nodes in DeepMind
    mesh-based simulation datasets.

    These labels are shared across multiple datasets (e.g. CylinderFlow, airfoils)
    and indicate whether a node represents fluid interior, boundaries, inflow/outflow regions, or solid obstacles.
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

def cells_to_bi_edges(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build unique bidirectional edges from triangle cells

    cells: shape (C, 3) int32
    returns senders, receivers: shape (E,) int32
    """
    a = cells[:, 0]
    b = cells[:, 1]
    c = cells[:, 2]

    e1 = np.stack([a, b], axis=1)
    e2 = np.stack([b, c], axis=1)
    e3 = np.stack([c, a], axis=1)
    edges = np.concatenate([e1, e2, e3], axis=0)      # (3C,2)

    rev = edges[:, ::-1]
    edges = np.concatenate([edges, rev], axis=0)      # (6C,2)

    # unique undirected+directed edges
    edges = np.unique(edges, axis=0)                  # (E,2)

    senders = edges[:, 0].astype(np.int32)
    receivers = edges[:, 1].astype(np.int32)
    return senders, receivers


def trajectory_iterator_np(tfrecord_path, meta_path, traj_ids=None):
    """
    Lazily stream and decode DeepMind CylinderFlow
    TFRecord trajectories into NumPy arrays.

    Each TFRecord entry corresponds to one full simulation trajectory.
    Features are decoded from raw bytes using shape and dtype information in `meta.json`,
    with the single `-1` dimension inferred from buffer size.
    Data are yielded one trajectory at a time without batching, caching, or tiling.

    Parameters
    ----------
    tfrecord_path : str or pathlib.Path
        Path to the TFRecord file.
    meta_path : str or pathlib.Path
        Path to the associated `meta.json`.

    Yields
    ------
    dict[str, np.ndarray]
        Feature arrays for a single trajectory
        (e.g. mesh_pos, node_type, velocity, pressure).
    """

    with open(meta_path, "r") as f:
        meta = json.load(f)

    data = tf.data.TFRecordDataset(tfrecord_path)
    feature_spec = { name: tf.io.FixedLenFeature([], tf.string) for name in meta["field_names"] }
    data = data.map(lambda x: tf.io.parse_single_example(x, feature_spec), num_parallel_calls=tf.data.AUTOTUNE)
    data = data.prefetch(tf.data.AUTOTUNE)

    traj_id = 0
    for rec in data:
        if traj_ids is not None and traj_id not in traj_ids:
            # print(f"Skip {traj_id}")
            traj_id += 1
            continue

        rec_bytes = {k: rec[k].numpy() for k in rec}

        decoded = {}
        for k,v in meta["features"].items():
            dtype = getattr(np, v["dtype"])
            flat = np.frombuffer(rec_bytes[k], dtype=dtype).copy()

            shape = list(v["shape"])
            known = 1
            uknown_id = None
            for i,s in enumerate(shape):
                if s == -1:
                    uknown_id = i
                else:
                    known *= s
            if uknown_id == None:
                raise ValueError("No -1 placeholder for number of nodes found")
            shape[uknown_id] = flat.size // known
            arr = flat.reshape(shape)

            decoded[k] = arr

        senders, receivers = cells_to_bi_edges(decoded["cells"][0])
        decoded["senders"] = senders
        decoded["receivers"] = receivers

        traj_id += 1
        yield decoded

def threaded_trajectory_iterator(
    tfrecord_path,
    meta_path,
    traj_ids=None,
    max_prefetch=4,
):
    """
    Producer–consumer wrapper around trajectory_iterator_np.

    Yields
    ------
    (dict | None, bool)
        (trajectory, is_new_epoch)
    """

    q = queue.Queue(maxsize=max_prefetch)

    def producer():
        for traj in trajectory_iterator_np(tfrecord_path, meta_path, traj_ids):
            q.put(traj)
        q.put(None)

    threading.Thread(
        target=producer,
        daemon=True,
    ).start()

    while True:
        yield q.get()

