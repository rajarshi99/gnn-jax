import json
import tensorflow as tf
import numpy as np

def trajectory_iterator_np(tfrecord_path, meta_path):
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
                print(k, shape)
                exit()
            shape[uknown_id] = flat.size // known
            print(k, shape)
            arr = flat.reshape(shape)

            decoded[k] = arr

        yield decoded
