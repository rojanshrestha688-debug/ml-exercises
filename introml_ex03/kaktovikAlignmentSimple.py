'''
Created on 20.06.2025

@author: Linda Schneider
'''

import numpy as np
import cv2

# do not import more modules!
# Use OpenCV only for basic image operations such as resizing and thresholding.
# Use NumPy for the bounding box computation and for centering the symbol.
# Do not use contour detection or connected components here.


def simpleAlignment(img, size=128):
    """
    Align a grayscale symbol by centering its foreground on a fixed canvas.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    # Step 1: Resize the input image to a fixed square size.
    # Allowed: cv2.resize.
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Step 2: Binarize the resized image with Otsu thresholding.
    # Allowed: cv2.threshold with Otsu.
    _, mask = cv2.threshold(
        resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 3: Find the bounding box of the foreground with NumPy only.
    # Hint: The symbols are dark, the background is bright.
    # Allowed: NumPy operations such as argwhere, min, max, slicing.
    # Not allowed: cv2.findContours, connectedComponents, or similar high-level localization helpers.
    coords = np.argwhere(mask > 0)

    # Fallback for an empty mask so the pipeline cannot crash.
    if coords.shape[0] == 0:
        return resized

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Step 4: Crop the grayscale region of interest from the resized image.
    # Use NumPy slicing.
    crop = resized[y_min:y_max + 1, x_min:x_max + 1]

    # Step 5: Resize the cropped region such that it fits into half the canvas.
    # Allowed: cv2.resize.
    target = size // 2
    crop_h, crop_w = crop.shape[:2]

    scale = target / float(max(crop_h, crop_w))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 6: Place the resized symbol in the center of a blank canvas.
    # Use NumPy indexing and array assignment for centering.
    canvas = np.full((size, size), 255, dtype=resized.dtype)

    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2

    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized_crop

    return canvas