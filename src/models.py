import os

import numpy as np
import pandas as pd
import tensorflow as tf


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


# Create a TensorFlow dataset from the mapping file
def create_dataset_from_mapping(
    mapping_file,
    root_dir=os.path.join(".", "..", "data"),
    files_dir=None,
):
    if files_dir is None:
        files_dir = os.path.abspath(os.path.join(root_dir, "features"))

    mapping = pd.read_csv(os.path.join(root_dir, mapping_file))
    asset_ids = mapping["asset_id"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices(asset_ids)
    dataset = dataset.map(
        lambda asset_id: load_feature_from_asset_id(asset_id, files_dir)
    )
    return dataset, len(asset_ids)
