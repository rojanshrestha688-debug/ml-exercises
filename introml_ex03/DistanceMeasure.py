'''
Distance measures for exercise 3.
'''

import numpy as np

# do not import more modules!
# Implement the formulas yourself using NumPy operations.
# Do not use external distance or metric libraries.


def mseDistance(imgA, imgB):
    """
    Compute the mean squared error between two equally sized grayscale images.
    """
    if imgA.shape != imgB.shape:
        raise ValueError("Images must have the same shape.")

    # TODO: implement the MSE formula yourself with NumPy operations.
    # Allowed: astype, subtraction, squaring, mean, or an explicit sum divided by the number of pixels.
    # Not allowed: external metric/distance helpers from scipy, sklearn, cv2, etc.

    # Cast to float to avoid overflow/wraparound on uint8 subtraction.
    a = imgA.astype(np.float64)
    b = imgB.astype(np.float64)

    diff = a - b
    mse = np.mean(diff * diff)

    return mse


def euclideanDistance(featureA, featureB):
    """
    Compute the Euclidean distance between two feature vectors.
    """
    if featureA.shape != featureB.shape:
        raise ValueError("Feature vectors must have the same shape.")

    # TODO: implement the Euclidean distance formula yourself with NumPy operations.
    # Allowed: subtraction, squaring, sum, and sqrt.
    # Not allowed: external metric/distance helpers from scipy, sklearn, cv2, etc.
    a = featureA.astype(np.float64)
    b = featureB.astype(np.float64)

    diff = a - b
    dist = np.sqrt(np.sum(diff * diff))

    return dist
