import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Eigenfaces import (
    calculate_feature_statistics,
    calculate_average_face,
    calculate_eigenfaces,
    classify_image,
    create_database_from_folder,
    get_feature_representation,
    process_and_train,
    reconstruct_image,
    standardize_features,
    train_both_classifiers,
)


def _make_pattern_image(pattern_name, rng, image_size=(24, 24)):
    img = np.zeros(image_size, dtype=np.uint8)

    if pattern_name == "vertical":
        img[:, 9:15] = 255
    elif pattern_name == "horizontal":
        img[9:15, :] = 255
    elif pattern_name == "cross":
        img[:, 10:14] = 255
        img[10:14, :] = 255
    else:
        raise ValueError(f"Unknown pattern name: {pattern_name}")

    noise = rng.normal(0, 15, image_size)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)


class TestEigenfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="introml_ex4_"))
        cls.dataset_dir = cls.temp_dir / "toy_dataset"
        cls.dataset_dir.mkdir()

        rng = np.random.default_rng(0)
        for class_name, pattern_name in (
            ("class_vertical", "vertical"),
            ("class_horizontal", "horizontal"),
            ("class_cross", "cross"),
        ):
            class_dir = cls.dataset_dir / class_name
            class_dir.mkdir()
            for index in range(12):
                image = _make_pattern_image(pattern_name, rng)
                cv2.imwrite(str(class_dir / f"sample_{index:02d}.png"), image)

        labels, data, _, cls.h, cls.w = create_database_from_folder(cls.dataset_dir, image_size=(24, 24))
        cls.labels = labels
        cls.data = data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            data,
            labels,
            test_size=0.25,
            random_state=0,
            stratify=labels,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_create_database_from_folder(self):
        labels, data, num_images, h, w = create_database_from_folder(self.dataset_dir, image_size=(24, 24))
        self.assertEqual(num_images, 36)
        self.assertEqual(data.shape, (36, 24 * 24))
        self.assertEqual((h, w), (24, 24))
        self.assertEqual(set(labels), {"class_vertical", "class_horizontal", "class_cross"})

    def test_average(self):
        a = np.eye(10, 20)
        b = 5 * np.eye(20, 10)
        true_a = np.array([0.1] * 10 + [0.0] * 10)
        true_b = np.array([0.25] * 10)

        avg_a = calculate_average_face(a)
        avg_b = calculate_average_face(b)

        self.assertEqual(avg_a.shape, true_a.shape)
        self.assertTrue(np.allclose(avg_a, true_a, atol=0.1))
        self.assertEqual(avg_b.shape, true_b.shape)
        self.assertTrue(np.allclose(avg_b, true_b, atol=0.1))

    def test_calculate_eigenfaces(self):
        train = np.arange(72, dtype=np.float64).reshape(8, 9)
        avg = np.mean(train, axis=0)
        expected = np.linalg.svd(train - avg, full_matrices=False)[2][:4]
        result = calculate_eigenfaces(train, avg, 4)

        self.assertEqual(result.shape, (4, 9))
        self.assertTrue(np.allclose(result @ result.T, np.eye(4), atol=1e-8))
        self.assertTrue(np.allclose(np.abs(result), np.abs(expected), atol=1e-8))

    def test_get_feature_representation(self):
        images = np.array([[5, 5, 5, 5], [3, 3, 3, 3]], dtype=np.float64)
        eigenfaces = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float64)
        avg = np.array([1, 1, 1, 1], dtype=np.float64)

        expected = (images - avg) @ eigenfaces[:2].T
        result = get_feature_representation(images, eigenfaces, avg, 2)

        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.allclose(result, expected))

    def test_calculate_feature_statistics(self):
        features = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])
        feature_mean, feature_std = calculate_feature_statistics(features)
        self.assertTrue(np.allclose(feature_mean, np.array([3.0, 4.0])))
        self.assertTrue(np.allclose(feature_std, np.array([np.std([1.0, 3.0, 5.0]), 1.0])))

    def test_standardize_features(self):
        features = np.array([[1.0, 10.0], [3.0, 10.0]])
        feature_mean = np.array([2.0, 10.0])
        feature_std = np.array([1.0, 1.0])
        standardized = standardize_features(features, feature_mean, feature_std)
        expected = np.array([[-1.0, 0.0], [1.0, 0.0]])
        self.assertTrue(np.allclose(standardized, expected))

    def test_reconstruct_image(self):
        avg = np.ones(12, dtype=np.float64)
        eigenfaces = np.zeros((3, 12), dtype=np.float64)
        reconstruction = reconstruct_image(np.arange(12), eigenfaces, avg, 3, 3, 4)
        self.assertEqual(reconstruction.shape, (3, 4))
        self.assertTrue(np.allclose(reconstruction, np.ones((3, 4))))

        image = np.array([2.0, 0.0, 0.0, 0.0])
        avg = np.zeros(4, dtype=np.float64)
        eigenfaces = np.eye(4, dtype=np.float64)
        reconstruction = reconstruct_image(image, eigenfaces, avg, 4, 2, 2)
        self.assertTrue(np.allclose(reconstruction, image.reshape(2, 2)))

    def test_process_and_train_logistic(self):
        eigenfaces, num_eigenfaces, avg = process_and_train(
            self.y_train,
            self.X_train,
            self.X_train.shape[0],
            self.h,
            self.w,
            classifier_type="logistic",
            num_eigenfaces=6,
        )

        self.assertEqual(eigenfaces.shape, (6, self.h * self.w))
        self.assertEqual(num_eigenfaces, 6)
        self.assertEqual(avg.shape, (self.h * self.w,))

        predictions = []
        for image in self.X_test:
            predictions.append(
                classify_image(
                    image,
                    eigenfaces,
                    avg,
                    num_eigenfaces,
                    self.h,
                    self.w,
                    classifier_type="logistic",
                )[0]
            )

        self.assertGreaterEqual(accuracy_score(self.y_test, predictions), 0.95)

    def test_classify_image(self):
        avg = np.zeros(self.h * self.w, dtype=np.float64)
        eigenfaces = np.eye(self.h * self.w, dtype=np.float64)[:6]
        prediction = classify_image(
            self.X_test[0],
            eigenfaces,
            avg,
            6,
            self.h,
            self.w,
            classifier_type="logistic",
        )
        self.assertEqual(prediction.shape, (1,))

    def test_train_both_classifiers(self):
        eigenfaces, num_eigenfaces, avg = train_both_classifiers(
            self.y_train,
            self.X_train,
            self.X_train.shape[0],
            self.h,
            self.w,
            num_eigenfaces=6,
        )

        for classifier_type in ("logistic", "gaussian_nb"):
            predictions = []
            for image in self.X_test:
                predictions.append(
                    classify_image(
                        image,
                        eigenfaces,
                        avg,
                        num_eigenfaces,
                        self.h,
                        self.w,
                        classifier_type=classifier_type,
                    )[0]
                )
            self.assertGreaterEqual(accuracy_score(self.y_test, predictions), 0.95)


if __name__ == "__main__":
    unittest.main()
