'''
Histogram of Oriented Gradients utilities for exercise 3.
'''

import numpy as np
import cv2

# do not import more modules!
# You may use cv2.Sobel for the derivatives.
# Compute magnitudes, orientations, histogram binning, and block normalization yourself with NumPy.
# Do not use cv2.HOGDescriptor or any other ready-made HOG implementation.


def computeGradients(img):
    """
    Compute gradient magnitudes and unsigned orientations in degrees.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    # TODO: compute Sobel derivatives, magnitudes, and orientations.
    # Allowed: cv2.Sobel for the x/y derivatives and NumPy for the remaining computations.
    # Not allowed: any ready-made HOG or feature extraction implementation.

    img = img.astype(np.float64)

    # Reduce to a single channel if a colour image is passed.
    if img.ndim == 3:
        img = img.mean(axis=2)

    # Fine-scale central-difference derivatives ([-1, 0, 1]) with no smoothing,
    # which the paper reports as the best-performing gradient.
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx * gx + gy * gy)

    # Unsigned orientation folded into [0, 180). Use explicit folding (not
    # modulo) so that an angle of exactly/near 0 stays near 0 instead of
    # wrapping up towards 180 due to tiny negative float values.
    orientation = np.rad2deg(np.arctan2(gy, gx))
    orientation[orientation < 0.0] += 180.0
    orientation[orientation >= 180.0] -= 180.0

    return magnitude, orientation


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    # TODO: divide the image into cells and accumulate magnitudes into bins.
    # Use NumPy indexing/loops to implement the histogram accumulation yourself.
    # Do not call a library routine that directly computes cell histograms for HOG.

    H, W = magnitude.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    bin_width = 180.0 / num_bins

    hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float64)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            mag_cell = magnitude[y0:y0 + cell_size, x0:x0 + cell_size]
            ori_cell = orientation[y0:y0 + cell_size, x0:x0 + cell_size]

            # Hard assignment of each pixel to a bin index, with the top
            # boundary (== num_bins) wrapped back into the valid range.
            bin_idx = (ori_cell / bin_width).astype(np.int64)
            bin_idx[bin_idx >= num_bins] = num_bins - 1

            # Magnitude-weighted accumulation; np.add.at handles repeats.
            np.add.at(hist[cy, cx], bin_idx.ravel(), mag_cell.ravel())

    return hist


def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    # TODO: compute the final descriptor from your own cell histograms.
    # Implement the block normalization and concatenation yourself with NumPy.
    # Do not use cv2.HOGDescriptor, skimage.feature.hog, or similar helpers.

    magnitude, orientation = computeGradients(img)
    hist = buildCellHistograms(magnitude, orientation, cell_size, num_bins)

    n_cells_y, n_cells_x, _ = hist.shape

    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    if n_blocks_y <= 0 or n_blocks_x <= 0:
        return np.zeros(0, dtype=np.float64)

    blocks = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = hist[by:by + block_size, bx:bx + block_size, :].ravel()

            # L2 block normalization.
            norm = np.sqrt(np.sum(block * block) + eps * eps)
            block = block / norm

            blocks.append(block)

    descriptor = np.concatenate(blocks)
    return descriptor
