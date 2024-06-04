import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from umap.umap_ import UMAP


def plot_clusters_scatter_umap(features, cluster_labels, n_clusters):
    umap_model = UMAP(n_components=2)
    umap_features = umap_model.fit_transform(features)

    plt.figure(figsize=(8, 6))

    for cluster in range(n_clusters):
        cluster_data = umap_features[cluster_labels == cluster]
        plt.scatter(
            cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster + 1}"
        )

    plt.title("Clusters Visualization (UMAP)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.grid(
        True, linestyle="--", linewidth=0.5, color="gray"
    )  # Added grid for better visibility
    plt.tight_layout()
    plt.show()


def plot_silhouette_umap(features, cluster_labels):
    plt.figure(figsize=(8, 6))

    silhouette_avg = silhouette_score(features, cluster_labels)
    sample_silhouette_values = silhouette_samples(features, cluster_labels)

    y_lower = 10
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab10(
        np.arange(len(unique_clusters)) % len(plt.cm.tab10.colors)
    )

    for i, cluster in enumerate(unique_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels == cluster
        ]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=cluster_colors[i],
            edgecolor=cluster_colors[i],
            alpha=0.7,
            label=f"Cluster {cluster + 1}",
        )
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
        y_lower = y_upper + 10

    plt.title("Silhouette Plot (UMAP)")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        label=f"Average ({silhouette_avg:.3f})",
    )
    plt.yticks([])
    plt.legend()
    plt.grid(
        True, linestyle="--", linewidth=0.5, color="gray"
    )  # Added grid for better visibility
    plt.tight_layout()
    plt.show()


def plot_clusters_scatter_umap_3d(features, cluster_labels, n_clusters):
    umap_model = UMAP(n_components=3)
    umap_features = umap_model.fit_transform(features)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cluster_colors = plt.cm.tab10(np.arange(n_clusters) % len(plt.cm.tab10.colors))
    for cluster in range(n_clusters):
        cluster_data = umap_features[cluster_labels == cluster]
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            cluster_data[:, 2],
            label=f"Cluster {cluster + 1}",
            c=cluster_colors[cluster],
        )

    ax.set_title("Clusters Visualization (UMAP)")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.set_zlabel("UMAP Component 3")
    ax.legend()
    plt.show()


def get_original_images(asset_ids, cluster_labels, n_clusters, n_images_per_cluster=5):
    cluster_images = []
    custer_images_desc = []
    images_dir = os.path.join("..", "data", "galaxy_zoo")

    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_asset_ids = [asset_ids[i] for i in cluster_indices]

        cur_cluster_images = [
            os.path.join(images_dir, str(asset_id) + ".jpg")
            for asset_id in cluster_asset_ids
        ]
        cur_cluseter_images_desc = [
            f"Cluster {cluster} - {str(asset_id)}" for asset_id in cluster_asset_ids
        ]

        r = min(n_images_per_cluster, len(cluster_asset_ids))
        cluster_images += [plt.imread(img) for img in cur_cluster_images[:r]]
        custer_images_desc += cur_cluseter_images_desc[:r]

    return cluster_images, custer_images_desc


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


def load_image_from_asset_id(asset_id, files_dir):
    image_path = tf.strings.join(
        [files_dir, "/", tf.strings.as_string(asset_id), ".jpg"]
    )

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224], method="bilinear")
    return image, asset_id


# Create a TensorFlow dataset from the mapping file
def create_dataset_from_mapping_vgg(
    mapping_file,
    root_dir=os.path.join(os.path.dirname(__file__), "..", "data"),
    files_dir=None,
):
    if files_dir is None:
        files_dir = os.path.abspath(os.path.join(root_dir, "processed"))

    mapping = pd.read_csv(os.path.join(root_dir, mapping_file))
    asset_ids = mapping["asset_id"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices(asset_ids)
    dataset = dataset.map(
        lambda asset_id: load_image_from_asset_id(asset_id, files_dir)
    )
    return dataset
