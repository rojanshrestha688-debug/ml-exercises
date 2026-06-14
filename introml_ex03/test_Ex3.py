from pathlib import Path
import sys
import unittest
from collections import Counter

import cv2
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from DistanceMeasure import euclideanDistance, mseDistance
from HOGFeature import buildCellHistograms, calculateHOG, computeGradients
from kaktovikAlignmentSimple import simpleAlignment


IMG_DIR = MODULE_DIR / "img"
RETRIEVAL_DIR = IMG_DIR / "retrieval"


def load_gray(name):
    img = cv2.imread(str(IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {IMG_DIR / name}")
    return img


def load_retrieval_items():
    items = []
    for path in sorted(RETRIEVAL_DIR.glob("*.jpg")):
        label = path.stem.split("_")[0]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        items.append((label, path.name, img))
    return items


def translate_image(img, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), borderValue=255)


def rotate_image(img, angle):
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), borderValue=255)


def top_k_accuracy(items, distance_fn, k):
    correct = 0
    for i, (label, _, _) in enumerate(items):
        ranked = []
        for j, (other_label, other_name, _) in enumerate(items):
            if i == j:
                continue
            ranked.append((distance_fn(i, j), other_label, other_name))
        ranked.sort(key=lambda row: row[0])
        labels = [row[1] for row in ranked[:k]]
        correct += label in labels
    return correct / len(items)


class TestDistanceMeasures(unittest.TestCase):
    def test_mseDistance(self):
        img_a = np.zeros((3, 3), dtype=np.uint8)
        img_b = np.zeros((3, 3), dtype=np.uint8)
        img_c = np.full((3, 3), 10, dtype=np.uint8)

        self.assertAlmostEqual(mseDistance(img_a, img_b), 0.0, places=6)
        self.assertAlmostEqual(mseDistance(img_a, img_c), 100.0, places=6)

    def test_euclideanDistance(self):
        vec_a = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vec_c = np.array([0.0, 2.0, 2.0], dtype=np.float32)

        self.assertAlmostEqual(euclideanDistance(vec_a, vec_b), 0.0, places=6)
        self.assertAlmostEqual(euclideanDistance(vec_a, vec_c), 1.0, places=6)


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.img = load_gray("kaktovik-008_rot_0.jpg")

    def test_simpleAlignment(self):
        aligned = simpleAlignment(self.img, size=128)
        self.assertEqual(aligned.shape, (128, 128))
        self.assertTrue(np.issubdtype(aligned.dtype, np.uint8))

        center_patch = aligned[48:80, 48:80]
        self.assertGreater(np.count_nonzero(center_patch < 220), 100)

        repeated = simpleAlignment(self.img, size=128)
        self.assertTrue(np.array_equal(aligned, repeated))


class TestHOG(unittest.TestCase):
    def test_computeGradients(self):
        img = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
        magnitude, orientation = computeGradients(img)

        self.assertEqual(magnitude.shape, img.shape)
        self.assertEqual(orientation.shape, img.shape)
        self.assertGreater(np.mean(magnitude[:, 1:-1]), 0.0)
        self.assertTrue(np.all((orientation >= 0.0) & (orientation < 180.0)))
        self.assertLess(np.mean(np.abs(orientation[:, 1:-1])), 1.0)

    def test_buildCellHistograms(self):
        magnitude = np.ones((8, 8), dtype=np.float32)
        orientation = np.zeros((8, 8), dtype=np.float32)
        histograms = buildCellHistograms(magnitude, orientation, cell_size=4, num_bins=9)

        self.assertEqual(histograms.shape, (2, 2, 9))
        self.assertTrue(np.all(histograms[:, :, 0] > 0.0))
        self.assertTrue(np.allclose(histograms[:, :, 1:], 0.0))

    def test_calculateHOG(self):
        aligned = simpleAlignment(load_gray("kaktovik-012_rot_0.jpg"), size=128)
        feature = calculateHOG(aligned, cell_size=8, block_size=2, num_bins=9)

        self.assertEqual(feature.shape, (8100,))
        self.assertTrue(np.all(np.isfinite(feature)))
        self.assertGreater(np.linalg.norm(feature), 0.0)


class TestRobustness(unittest.TestCase):
    def setUp(self):
        self.same_writer = load_gray("kaktovik-012_rot_0.jpg")
        self.same_symbol = load_gray("kaktovik-008_rot_0.jpg")
        self.other_symbol = load_gray("kaktovik-001_rot_0.jpg")
        self.translated = translate_image(self.same_writer, 18, -10)
        self.rotated = rotate_image(self.same_writer, 18)

    def test_same_symbol_is_closer_than_other_symbol(self):
        same_symbol_distance = euclideanDistance(
            calculateHOG(simpleAlignment(self.same_writer)),
            calculateHOG(simpleAlignment(self.same_symbol)),
        )
        other_symbol_distance = euclideanDistance(
            calculateHOG(simpleAlignment(self.same_writer)),
            calculateHOG(simpleAlignment(self.other_symbol)),
        )
        self.assertLess(same_symbol_distance, other_symbol_distance)

    def test_translation_improves_after_alignment(self):
        aligned_original = simpleAlignment(self.same_writer)
        aligned_translated = simpleAlignment(self.translated)
        aligned_other = simpleAlignment(self.other_symbol)

        aligned_mse = mseDistance(aligned_original, aligned_translated)
        other_mse = mseDistance(aligned_original, aligned_other)
        self.assertLess(aligned_mse, other_mse)

        aligned_hog = euclideanDistance(calculateHOG(aligned_original), calculateHOG(aligned_translated))
        other_hog = euclideanDistance(calculateHOG(aligned_original), calculateHOG(aligned_other))
        self.assertLess(aligned_hog, other_hog)

    def test_rotation_remains_challenging(self):
        aligned_original = simpleAlignment(self.same_writer)
        aligned_rotated = simpleAlignment(self.rotated)

        translated_hog = euclideanDistance(calculateHOG(aligned_original),
                                           calculateHOG(simpleAlignment(self.translated)))
        rotated_hog = euclideanDistance(calculateHOG(aligned_original), calculateHOG(aligned_rotated))
        self.assertGreater(rotated_hog, translated_hog)


class TestRetrieval(unittest.TestCase):
    def test_gallery_exists(self):
        items = load_retrieval_items()
        counts = Counter(label for label, _, _ in items)

        self.assertEqual(len(items), 220)
        self.assertEqual(len(counts), 22)
        self.assertTrue(all(count == 10 for count in counts.values()))

    def test_hog_retrieval_is_strong_on_real_gallery(self):
        items = load_retrieval_items()
        aligned_images = [simpleAlignment(img) for _, _, img in items]
        aligned_hog = [calculateHOG(img) for img in aligned_images]
        aligned_top1 = top_k_accuracy(items, lambda i, j: euclideanDistance(aligned_hog[i], aligned_hog[j]), 1)
        aligned_top3 = top_k_accuracy(items, lambda i, j: euclideanDistance(aligned_hog[i], aligned_hog[j]), 3)
        aligned_top5 = top_k_accuracy(items, lambda i, j: euclideanDistance(aligned_hog[i], aligned_hog[j]), 5)

        self.assertGreaterEqual(aligned_top1, 0.70)
        self.assertGreaterEqual(aligned_top3, 0.84)
        self.assertGreaterEqual(aligned_top5, 0.90)

    def test_hog_outperforms_mse_after_alignment(self):
        items = load_retrieval_items()
        aligned_images = [simpleAlignment(img) for _, _, img in items]
        aligned_hog = [calculateHOG(img) for img in aligned_images]

        hog_top1 = top_k_accuracy(items, lambda i, j: euclideanDistance(aligned_hog[i], aligned_hog[j]), 1)
        hog_top3 = top_k_accuracy(items, lambda i, j: euclideanDistance(aligned_hog[i], aligned_hog[j]), 3)
        mse_top1 = top_k_accuracy(items, lambda i, j: mseDistance(aligned_images[i], aligned_images[j]), 1)
        mse_top3 = top_k_accuracy(items, lambda i, j: mseDistance(aligned_images[i], aligned_images[j]), 3)

        self.assertGreater(hog_top1, mse_top1)
        self.assertGreater(hog_top3, mse_top3)


if __name__ == "__main__":
    unittest.main()
