import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def encode(name, encoder, batch_size=256):
    dataset, _ = create_dataset_from_mapping(mapping_file=f"{name}_mapping.csv")
    dataset = dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    data_tensors = []
    label_tensors = []

    for data, label in tqdm(dataset, desc=f"Encoding {name}", total=len(dataset)):
        data = encoder(data)
        data_tensors.append(data)
        label_tensors.append(label)

    data_tensor = tf.concat(data_tensors, axis=0)
    label_tensor = tf.concat(label_tensors, axis=0)

    np.save(os.path.join(root_dir, "data", f"X_{name}_encoded.npy"), data_tensor)
    np.save(os.path.join(root_dir, "data", f"y_{name}.npy"), label_tensor)


def load_entire(dataset, size):
    data_tensors = []
    label_tensors = []

    for data, label in dataset.batch(size):
        data_tensors.append(data)
        label_tensors.append(label)

    data_tensor = tf.concat(data_tensors, axis=0)
    label_tensor = tf.concat(label_tensors, axis=0)
    return data_tensor, label_tensor


def load_feature_from_asset_id(asset_id, files_dir):
    feature_path = tf.strings.join(
        [files_dir, "/", tf.strings.as_string(asset_id), ".npy"]
    )

    feature = tf.numpy_function(np.load, [feature_path], tf.float32)
    feature = tf.convert_to_tensor(feature)
    feature = tf.reshape(feature, (1, 4, 4, 1536))
    feature = tf.nn.avg_pool(
        feature, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )
    feature = tf.reshape(feature, (4 * 1536,))
    return feature, asset_id


def load_feature_from_asset_id_2(asset_id, files_dir):
    feature_path = tf.strings.join(
        [files_dir, "/", tf.strings.as_string(asset_id), ".npy"]
    )

    feature = tf.numpy_function(np.load, [feature_path], tf.float32)
    feature = tf.convert_to_tensor(feature)
    feature = tf.reshape(feature, (1, 4, 4, 1536))
    feature = tf.nn.avg_pool(
        feature, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )
    feature = tf.reshape(feature, (4 * 1536,))
    return feature, feature


# Create a TensorFlow dataset from the mapping file
def create_dataset_from_mapping(
    mapping_file,
    root_dir=os.path.join(".", "..", "data"),
    files_dir=None,
    mode=1,
    max_size=10000,
):
    if files_dir is None:
        files_dir = os.path.abspath(os.path.join(root_dir, "features"))

    mapping = pd.read_csv(os.path.join(root_dir, mapping_file))
    asset_ids = mapping["asset_id"].tolist()

    if mode == 1:
        dataset = tf.data.Dataset.from_tensor_slices(asset_ids)
        dataset = dataset.map(
            lambda asset_id: load_feature_from_asset_id(asset_id, files_dir)
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(asset_ids[:max_size])
        dataset = dataset.map(
            lambda asset_id: load_feature_from_asset_id_2(asset_id, files_dir)
        )

    return dataset, len(asset_ids)
