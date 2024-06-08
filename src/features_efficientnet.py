import multiprocessing
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
warnings.simplefilter("ignore")

from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tqdm import tqdm

_MEAN = [24.008411, 30.535658, 34.82993]
_STD = [32.271805, 38.59381, 45.358646]
MEAN = tf.constant(_MEAN, shape=(1, 1, 3), dtype=tf.float32)
STD = tf.constant(_STD, shape=(1, 1, 3), dtype=tf.float32)


def get_transform(size=None):
    def transform(image):
        # Normalize the image
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - MEAN) / STD

        if size is not None:
            image = tf.image.resize(image, [size, size])

        return image

    return transform


def load_image_from_asset_id(asset_id, files_dir, transform=None):
    image_path = tf.strings.join(
        [files_dir, "/", tf.strings.as_string(asset_id), ".jpg"]
    )

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    if transform:
        image = transform(image)
    return image, asset_id


def create_dataset_from_mapping(
    mapping_file,
    root_dir=os.path.join(os.path.dirname(__file__), "..", "data"),
    files_dir=None,
    transform=None,
):
    if files_dir is None:
        files_dir = os.path.abspath(os.path.join(root_dir, "processed"))

    mapping = pd.read_csv(os.path.join(root_dir, mapping_file))
    asset_ids = mapping["asset_id"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices(asset_ids)
    dataset = dataset.map(
        lambda asset_id: load_image_from_asset_id(
            asset_id, files_dir, transform=transform
        )
    )
    return dataset


def compute_mean_std(dataset):
    mean_sum = tf.zeros((3,), dtype=tf.float32)
    sq_diff_sum = tf.zeros((3,), dtype=tf.float32)
    total_images = tf.constant(0, dtype=tf.float32)

    for images, _ in dataset:
        batch_size = tf.cast(tf.shape(images)[0], tf.float32)
        total_images += batch_size
        images = tf.cast(images, tf.float32)
        mean_sum += tf.reduce_sum(tf.reduce_mean(images, axis=[1, 2]), axis=0)
        sq_diff_sum += tf.reduce_sum(
            tf.reduce_mean(tf.square(images), axis=[1, 2]), axis=0
        )

    mean = mean_sum / total_images
    std = tf.sqrt(sq_diff_sum / total_images - tf.square(mean))

    return mean, std


def reverse_transform(tensor):
    tensor = tensor * STD + MEAN
    return tensor.numpy()[0]


def preprocess_image_2(image):
    return feature_extractor.predict(image)


def _process_helper(dt, output_directory):
    feature, img_id = dt
    filepath = os.path.join(output_directory, f"{img_id}.npy")
    with open(filepath, "wb") as f:
        np.save(f, feature.flatten())


def _process(name, batch_size=128):
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using device: {physical_devices[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU device found, using CPU")

    dataset = create_dataset_from_mapping(mapping_file=name, transform=get_transform())
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    root_dir = os.path.abspath(os.path.join(__file__, "..", "..", "data"))
    output_directory = os.path.join(root_dir, "features")

    base_model = EfficientNetB3(weights="imagenet", include_top=False)
    feature_extractor = tf.keras.Model(
        inputs=base_model.input, outputs=base_model.output
    )
    helper = partial(
        _process_helper,
        output_directory=output_directory,
    )

    os.makedirs(output_directory, exist_ok=True)
    pool = multiprocessing.Pool(2)

    dataloader = tqdm(dataset, unit="batch")
    for batch_idx, (data, asset_ids) in enumerate(dataloader):
        features = feature_extractor.predict(data)
        pool.map(helper, zip(features, asset_ids))

    pool.close()
    pool.join()


def _main():
    if "--name" in sys.argv:
        name_index = sys.argv.index("--name") + 1
        if name_index < len(sys.argv):
            name = sys.argv[name_index]
            _process(name)
        else:
            print("Error: No name provided after --name")
            exit(-1)
    else:
        print("Error: --name argument not provided")
        exit(-1)


if __name__ == "__main__":
    _main()
