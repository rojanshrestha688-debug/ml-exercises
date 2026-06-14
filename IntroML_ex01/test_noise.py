'''Unit tests for the noise functions.
These tests verify that the implemented noise generators produce
outputs with the correct shape, dtype and value ranges.
''' 

import unittest
import os
import numpy as np
from pathlib import Path
from noise import (
    load_image,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    add_poisson_noise,
    add_uniform_noise,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

class TestNoiseFunctions(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure the test image exists
        assert os.path.exists(DATA_DIR / 'hello.png'), 'hello.png missing'
        self.img = load_image(str(DATA_DIR / 'hello.png'))
        self.assertEqual(self.img.dtype, np.uint8)
        self.assertTrue(self.img.ndim in (2, 3))

    def _check_common(self, noisy):
        self.assertIsInstance(noisy, np.ndarray)
        self.assertEqual(noisy.shape, self.img.shape)
        self.assertEqual(noisy.dtype, np.uint8)
        self.assertTrue(noisy.min() >= 0 and noisy.max() <= 255)

    def test_gaussian_noise(self):
        noisy = add_gaussian_noise(self.img)
        self._check_common(noisy)
        # Expect at least one pixel to differ from the original
        self.assertFalse(np.array_equal(noisy, self.img))

    def test_salt_and_pepper_noise(self):
        noisy = add_salt_and_pepper_noise(self.img, salt_prob=0.05, pepper_prob=0.05)
        self._check_common(noisy)
        # Check that there are some 0 and 255 values introduced
        self.assertTrue((noisy == 0).any(), 'No pepper (0) pixels added')
        self.assertTrue((noisy == 255).any(), 'No salt (255) pixels added')

    def test_poisson_noise(self):
        noisy = add_poisson_noise(self.img)
        self._check_common(noisy)
        self.assertFalse(np.array_equal(noisy, self.img))

    def test_uniform_noise(self):
        noisy = add_uniform_noise(self.img)
        self._check_common(noisy)
        self.assertFalse(np.array_equal(noisy, self.img))

if __name__ == '__main__':
    unittest.main()
