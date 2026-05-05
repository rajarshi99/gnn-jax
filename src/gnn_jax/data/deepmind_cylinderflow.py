import json
import tensorflow as tf
import numpy as np
import enum

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

def trajectory_iterator_np(tfrecord_path, meta_path):
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
    data = data.map(lambda x: tf.io.parse_single_example(x, feature_spec), num_parallel_calls=1)

    for rec in data:
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
                print("No -1 placeholder for number of nodes found", k, shape)
                exit()
            shape[uknown_id] = flat.size // known
            arr = flat.reshape(shape)

            decoded[k] = arr

        yield decoded
