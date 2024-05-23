import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from preprocessing import calculate_threshold, convert_to_grayscale, find_central_source

warnings.simplefilter("ignore")
np.random.seed(42)


def load_images(filepaths):
    images = []
    for filepath in filepaths:
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.stack(images)


def show_images(images, titles=None, ncols=4, figsize=None, **kwargs):
    nrows = len(images) // ncols
    if len(images) % ncols != 0:
        nrows += 1
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image, **kwargs)
        if titles is not None:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def show_pixels(images, images_per_class, n_classes, class_names=None, figsize=None):
    colors = ["red", "green", "blue"]
    ncols = 3
    nrows = n_classes
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx in range(n_classes):
        for image in images[idx : idx + images_per_class]:
            for j in range(3):
                ax = axes[ncols * idx + j]
                sns.histplot(
                    image[:, :, j].flatten(),
                    ax=ax,
                    bins=50,
                    color=colors[j],
                    alpha=1.0 / images_per_class,
                )

                if j == 1 and class_names is not None:
                    ax.set_title(class_names[idx])

    plt.tight_layout()
    plt.show()


def show_images_layers(images, titles=None, figsize=None, gray_=True):
    colors = ["Reds", "Greens", "Blues"]
    ncols = 5 if gray_ else 4
    nrows = len(images)
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, image in enumerate(images):
        for j in range(3):
            ax = axes[ncols * i + j]
            ax.imshow(image[:, :, j], cmap=colors[j])

        ax = axes[ncols * i + 3]
        if titles is not None:
            ax.set_title(titles[i])
        ax.imshow(image)

        if gray_:
            ax = axes[ncols * i + 4]
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ax.imshow(image_gray, cmap="Greys")

    plt.tight_layout()
    plt.show()


def get_binary_image(image, sigma=4):
    gray_image = convert_to_grayscale(image)
    threshold = calculate_threshold(gray_image, sigma=sigma)

    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def get_boxes(image, sigma=4):
    gray_image = convert_to_grayscale(image)
    threshold = calculate_threshold(gray_image, sigma=sigma)

    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    central_contour = find_central_source(gray_image, threshold)

    hull = cv2.convexHull(central_contour)
    cv2.polylines(image, [hull], True, (0, 255, 0), 4)
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    return image
