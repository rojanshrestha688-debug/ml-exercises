'''
Unit tests for the histogram equalization skeleton.
The student must implement the functions so that these tests pass.
The tests reflect the notes given in the exercise sheet.
'''

import unittest
import cv2
import numpy as np
import os
from pathlib import Path
from histogram_equalization import compute_histogram, compute_cdf, equalize_image, load_image

# 5% tolerance for integer counts, tighter for floating point
RTOL_INT = 0.05
RTOL_FLOAT = 1e-6
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

class TestHistogramEqualization(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure the working directory contains the data folder
        assert os.path.exists(DATA_DIR / 'hello.png'), "hello.png not found"
        self.img = load_image(str(DATA_DIR / 'hello.png'))
        assert self.img.dtype == np.uint8

    def test_compute_histogram(self):
        hist = compute_histogram(self.img)
        self.assertIsInstance(hist, np.ndarray, msg='Histogram must be a NumPy array')
        self.assertEqual(hist.shape[0], 256, msg='Histogram must have 256 bins')
        # The exercise sheet states that the sum of the first 90 bins should be 249
        self.assertAlmostEqual(np.sum(hist[:90]), 249, delta=RTOL_INT * 249,
                               msg='First 90 histogram bins sum mismatch')

    def test_compute_cdf(self):
        hist = compute_histogram(self.img)
        cdf = compute_cdf(hist)
        self.assertIsInstance(cdf, np.ndarray, msg='CDF must be a NumPy array')
        self.assertEqual(cdf.shape[0], 256, msg='CDF must have 256 entries')
        # The exercise sheet gives the first value of the cumulative distribution
        # (sum of first 90 entries) as approximately 0.001974977
        self.assertAlmostEqual(np.sum(cdf[:90]), 0.001974977, delta=RTOL_FLOAT,
                               msg='First 90 CDF sum mismatch')

    def test_equalize_image_shape_and_type(self):
        hist = compute_histogram(self.img)
        cdf = compute_cdf(hist)
        eq_img = equalize_image(self.img, cdf)
        self.assertIsInstance(eq_img, np.ndarray)
        self.assertEqual(eq_img.shape, self.img.shape, msg='Equalized image shape must match original')
        self.assertEqual(eq_img.dtype, np.uint8, msg='Equalized image must be uint8')

if __name__ == '__main__':
    unittest.main()
