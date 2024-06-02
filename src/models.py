import os

import pandas as pd
import tensorflow as tf

_MEAN = [24.008411, 30.535658, 34.82993]
_STD = [32.271805, 38.59381, 45.358646]
MEAN = tf.constant(_MEAN, shape=(1, 1, 3), dtype=tf.float32)
STD = tf.constant(_STD, shape=(1, 1, 3), dtype=tf.float32)


def get_transform(size=None, train=False):
    def transform(image):
        # Normalize the image
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - MEAN) / STD

        if train:
            # Apply color jitter
            def color_jitter(image):
                image = tf.image.random_brightness(image, max_delta=0.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.1)
                return image

            image = tf.cond(
                tf.random.uniform([]) < 0.8, lambda: color_jitter(image), lambda: image
            )

            # Apply random rotation
            angle = tf.random.uniform([], minval=-180, maxval=180, dtype=tf.float32)
            radians = angle * 3.141592653589793 / 180.0
            image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))

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
    return image


# Create a TensorFlow dataset from the mapping file
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

    for images in dataset:
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
