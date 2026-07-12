import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import confusion_matrix
from knn import KNNClassifier
from logistic_regression import LogisticRegressionClassifier
from nbnn import NBNNClassifier
from main import IMG_SIZE, TRAIN_DIR, UNIT_TEST_DIR, load_images_labels
from visualization import plot_confusion_matrix, plot_knn_neighbors


class TestExercise5(unittest.TestCase):
    def _skip_if_no_sklearn(self):
        try:
            classifier = LogisticRegressionClassifier()
            classifier.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        except ImportError:
            self.skipTest("scikit-learn is not installed in this environment.")

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="introml_ex5_"))

        cls.X_train_real, cls.y_train_real = load_images_labels(TRAIN_DIR)
        cls.X_unit_real, cls.y_unit_real = load_images_labels(UNIT_TEST_DIR)

        classes = np.unique(np.concatenate([cls.y_train_real, cls.y_unit_real]))
        cls.class_to_int = {label: index for index, label in enumerate(classes)}
        cls.y_train_real_enc = np.asarray(
            [cls.class_to_int[label] for label in cls.y_train_real],
            dtype=int,
        )
        cls.y_unit_real_enc = np.asarray(
            [cls.class_to_int[label] for label in cls.y_unit_real],
            dtype=int,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_fit_stores_training_data(self):
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        y = ["a", "b", "c"]

        classifier = KNNClassifier(n_neighbors=1)
        returned = classifier.fit(X, y)

        self.assertIs(returned, classifier)
        self.assertIsInstance(classifier.X_train, np.ndarray)
        self.assertIsInstance(classifier.y_train, np.ndarray)
        self.assertEqual(classifier.X_train.shape, (3, 2))
        self.assertEqual(classifier.y_train.shape, (3,))

    def test_predict_raises_before_fit(self):
        classifier = KNNClassifier(n_neighbors=1)
        with self.assertRaises(ValueError):
            classifier.predict(np.array([[0.0, 0.0]]))

    def test_predict_euclidean_distance(self):
        X_train = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [3.0, 3.0],
                [3.0, 4.0],
            ]
        )
        y_train = np.array([0, 0, 0, 1, 1])
        X_test = np.array([[0.1, 0.2], [3.2, 3.1]])

        classifier = KNNClassifier(n_neighbors=3, metric="euclidean")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_predict_cosine_distance_handles_zero_vectors(self):
        X_train = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        y_train = np.array([0, 1, 2])
        X_test = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 0.0],
            ]
        )

        classifier = KNNClassifier(n_neighbors=1, metric="cosine")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        np.testing.assert_array_equal(predictions[:2], np.array([0, 1]))
        self.assertIn(predictions[2], y_train)

    def test_tie_breaking_is_deterministic(self):
        X_train = np.array([[0.0], [0.1], [0.2], [0.3]])
        y_train = np.array([2, 0, 2, 0])

        classifier = KNNClassifier(n_neighbors=4, metric="euclidean")
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(np.array([[0.15]]))

        self.assertEqual(prediction[0], 0)

    def test_confusion_matrix_counts_correctly(self):
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 2, 1, 1, 0, 2])
        expected = np.array(
            [
                [2, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
            ]
        )

        cm = confusion_matrix(y_true, y_pred)

        np.testing.assert_array_equal(cm, expected)

    def test_confusion_matrix_infers_matrix_size(self):
        y_true = np.array([0, 2, 2])
        y_pred = np.array([2, 1, 2])
        expected = np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 1, 1],
            ]
        )

        cm = confusion_matrix(y_true, y_pred)

        self.assertEqual(cm.shape, (3, 3))
        np.testing.assert_array_equal(cm, expected)

    def test_nbnn_predicts_nearest_class(self):
        X_train = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [3.0, 3.0],
                [3.2, 3.1],
            ]
        )
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[0.1, 0.0], [3.1, 3.2]])

        classifier = NBNNClassifier(metric="euclidean")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_nbnn_handles_cosine_distance(self):
        X_train = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]
        )
        y_train = np.array([0, 0, 1, 1])

        classifier = NBNNClassifier(metric="cosine")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(np.array([[0.9, 0.1], [0.1, 0.9]]))

        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_load_images_labels_uses_new_kaktovik_data(self):
        self.assertEqual(self.X_train_real.shape, (2882, IMG_SIZE[0] * IMG_SIZE[1]))
        self.assertEqual(self.X_unit_real.shape, (11, IMG_SIZE[0] * IMG_SIZE[1]))
        self.assertEqual(len(np.unique(self.y_train_real)), 11)
        self.assertEqual(set(np.unique(self.y_unit_real)), set(np.unique(self.y_train_real)))

    def test_knn_reaches_high_accuracy_on_unit_test_images(self):
        thresholds = {
            "euclidean": 0.55,
            "cosine": 0.55,
        }

        for metric, threshold in thresholds.items():
            classifier = KNNClassifier(n_neighbors=3, metric=metric)
            classifier.fit(self.X_train_real, self.y_train_real_enc)
            predictions = classifier.predict(self.X_unit_real)
            accuracy = np.mean(predictions == self.y_unit_real_enc)
            self.assertGreaterEqual(accuracy, threshold)

    def test_nbnn_reaches_reasonable_accuracy_on_unit_test_images(self):
        thresholds = {
            "euclidean": 0.55,
            "cosine": 0.55,
        }

        for metric, threshold in thresholds.items():
            classifier = NBNNClassifier(metric=metric)
            classifier.fit(self.X_train_real, self.y_train_real_enc)
            predictions = classifier.predict(self.X_unit_real)
            accuracy = np.mean(predictions == self.y_unit_real_enc)
            self.assertGreaterEqual(accuracy, threshold)

    def test_logistic_regression_smoke(self):
        self._skip_if_no_sklearn()

        classifier = LogisticRegressionClassifier(max_iter=2000, random_state=0)
        classifier.fit(self.X_train_real, self.y_train_real_enc)
        predictions = classifier.predict(self.X_unit_real)
        accuracy = np.mean(predictions == self.y_unit_real_enc)

        self.assertGreaterEqual(accuracy, 0.55)

    def test_logistic_regression_predict_raises_before_fit(self):
        self._skip_if_no_sklearn()

        classifier = LogisticRegressionClassifier()
        with self.assertRaises(ValueError):
            classifier.predict(np.array([[0.0, 0.0]]))

    def test_logistic_regression_toy_example(self):
        self._skip_if_no_sklearn()

        X_train = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [3.0, 3.0],
                [3.0, 4.0],
                [4.0, 3.0],
            ]
        )
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[0.2, 0.1], [3.5, 3.2]])

        classifier = LogisticRegressionClassifier(max_iter=2000, random_state=0)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_visualizations_can_be_saved(self):
        neighbor_path = self.temp_dir / "neighbors.png"
        confusion_path = self.temp_dir / "confusion.png"

        plot_knn_neighbors(
            test_image=np.zeros((8, 8)),
            neighbor_images=[np.ones((8, 8)), np.full((8, 8), 0.5)],
            neighbor_labels=[0, 1],
            pred_label=0,
            save_path=neighbor_path,
        )
        plot_confusion_matrix(
            np.array([[2, 0], [1, 3]]),
            ["0", "1"],
            title="Toy Confusion Matrix",
            save_path=confusion_path,
        )

        self.assertTrue(neighbor_path.exists())
        self.assertTrue(confusion_path.exists())


if __name__ == "__main__":
    unittest.main()
