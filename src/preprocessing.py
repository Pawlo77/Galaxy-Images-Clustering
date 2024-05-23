import multiprocessing
import os
import sys
import warnings
from functools import partial

import cv2
import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats

warnings.simplefilter("ignore")
np.random.seed(42)


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calculate_threshold(image_data, sigma=4):
    mean, median, std = sigma_clipped_stats(image_data, sigma=sigma)
    threshold = median + (sigma * std)
    return threshold


def find_central_source(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    def contour_distance_from_center(contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt(
                (cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2
            )
            return int(distance)
        return float("inf")

    if len(contours) == 0:
        return None

    return max(
        contours,
        key=lambda c: cv2.contourArea(c) - contour_distance_from_center(c) ** 2,
    )


def clear_image(image, c, plot=False):
    hull = cv2.convexHull(c)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))

    if plot:
        cv2.polylines(image, [hull], True, (0, 255, 0), 2)
    return cv2.bitwise_and(image, mask)


def map_image_threshold(image_data, threshold):
    return np.where(image_data < threshold, 0, image_data)


def extract_and_resize(image, c, target_size=(128, 128), return_size=False, plot=False):
    height, width = image.shape[:2]

    rect = cv2.minAreaRect(c)
    (center_x, center_y), (w, h), angle = rect

    image = clear_image(image, c, plot=plot)

    max_side = max(w, h)
    w = h = max_side

    rect = ((center_x, center_y), (w, h), angle)

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))

    box = cv2.boxPoints(rect)
    box = np.int0(cv2.transform(np.array([box]), rotation_matrix)[0])

    x, y, w, h = cv2.boundingRect(box)
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if return_size:
        return w, h

    source_region = image[y : y + h, x : x + w]
    resized_source = cv2.resize(
        source_region, target_size, interpolation=cv2.INTER_AREA
    )
    return resized_source


def preprocess_image(
    image,
    sigma=4,
    resize_factor=2,
    target_size=(128, 128),
    plot=False,
    return_size=False,
):
    gray_image = convert_to_grayscale(image)
    threshold = calculate_threshold(gray_image, sigma=sigma)
    central_contour = find_central_source(gray_image, threshold)

    return extract_and_resize(
        image, central_contour, target_size, return_size=return_size, plot=plot
    )


def extract_and_resize_2(
    image,
    c,
    target_size=(128, 128),
    return_size=False,
    plot=False,
    min_size=120,
    reset_size=200,
    plot_box=False,
):
    height, width = image.shape[:2]

    if c is not None:
        rect = cv2.minAreaRect(c)
        (center_x, center_y), (w, h), angle = rect
    else:
        center_x, center_y = width // 2, height // 2
        w = h = angle = 0

    if plot_box:
        cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

    max_side = max(w, h)
    if max_side > min_size:
        image = clear_image(image, c, plot=plot)
    else:
        center_x, center_y = width // 2, height // 2
        angle = 0
        max_side = reset_size

    w = h = max_side
    rect = ((center_x, center_y), (w, h), angle)
    box = cv2.boxPoints(rect)

    if plot_box:
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
        cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        return image

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))

    box = np.int0(cv2.transform(np.array([box]), rotation_matrix)[0])
    x, y, w, h = cv2.boundingRect(box)
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if return_size:
        return w, h

    source_region = image[y : y + h, x : x + w]
    resized_source = cv2.resize(
        source_region, target_size, interpolation=cv2.INTER_AREA
    )
    return resized_source


def preprocess_image_2(
    image,
    sigma=4,
    target_size=(128, 128),
    plot=False,
    return_size=False,
    plot_box=False,
    min_size=100,
    reset_size=200,
):
    gray_image = convert_to_grayscale(image)
    threshold = calculate_threshold(gray_image, sigma=sigma)
    central_contour = find_central_source(gray_image, threshold)

    return extract_and_resize_2(
        image,
        central_contour,
        target_size,
        min_size=min_size,
        reset_size=reset_size,
        return_size=return_size,
        plot=plot,
        plot_box=plot_box,
    )


def _process_helper(img_id, image_directory, output_directory):
    filepath = os.path.join(image_directory, f"{img_id}.jpg")
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_image_2(image, sigma=4, target_size=(150, 150))
    filepath = os.path.join(output_directory, f"{img_id}.jpg")
    cv2.imwrite(filepath, image)


def _process(name):
    root_dir = os.path.abspath(os.path.join(__file__, "..", "..", "data"))
    mapping = pd.read_csv(os.path.join(root_dir, name))
    image_directory = os.path.join(root_dir, "galaxy_zoo")
    output_directory = os.path.join(root_dir, "processed")

    os.makedirs(output_directory, exist_ok=True)

    helper = partial(
        _process_helper,
        image_directory=image_directory,
        output_directory=output_directory,
    )
    pool = multiprocessing.Pool()
    pool.map(helper, mapping["asset_id"])
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
