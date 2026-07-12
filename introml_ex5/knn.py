from pathlib import Path

import numpy as np

from visualization import plot_knn_neighbors


class KNNClassifier:
    def __init__(
        self,
        n_neighbors=3,
        metric="euclidean",
        plot_neighbors=False,
        image_shape=None,
        plot_dir=None,
        plot_prefix="knn",
    ):
        # Number of nearest neighbours to use for prediction.
        self.n_neighbors = n_neighbors
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.plot_neighbors = plot_neighbors
        self.image_shape = image_shape
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - check that X has shape (n_samples, n_features)
            - check that len(X) == len(y)
            - return self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        diff = self.X_train - x
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Hint:
            cosine_distance = 1 - cosine_similarity
            cosine_similarity = (a · b) / (||a|| * ||b||)

        Make sure that zero vectors do not cause a division-by-zero error.
        """
        train_norms = np.linalg.norm(self.X_train, axis=1)
        x_norm = np.linalg.norm(x)

        denom = train_norms * x_norm
        denom = np.where(denom == 0, 1e-10, denom)

        dot_products = self.X_train @ x
        cosine_similarity = dot_products / denom

        return 1 - cosine_similarity

    def _majority_vote(self, neighbor_labels):
        """
        Return the most frequent label among the nearest neighbours.

        Hint:
            np.unique(..., return_counts=True) is useful here.
            If there is a tie, choose the smallest label after sorting.
        """
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = counts.max()
        tied_labels = labels[counts == max_count]
        return tied_labels.min()

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - convert X to a NumPy array
            - allow a single sample with shape (n_features,)
            - compute distances with the selected metric
            - select the k nearest neighbours
            - predict by majority vote
            - optionally save neighbour plots when plot_neighbors is True
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("This KNNClassifier instance is not fitted yet. Call fit() first.")

        # Convert input data to a NumPy array.
        X = np.asarray(X)

        # If a single sample is passed, turn it into shape (1, n_features).
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        # Iterate over each test sample to predict its label.
        for sample_index, x in enumerate(X):
            if self.metric == "euclidean":
                # Compute Euclidean distances from x to all training samples.
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                # Compute cosine distances from x to all training samples.
                distances = self._cosine_distances(x)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            # Find the indices of the k nearest neighbours.
            k = min(self.n_neighbors, len(self.X_train))
            nn_indices = np.argsort(distances, kind="stable")[:k]

            # Get the labels of the nearest neighbours.
            nn_labels = self.y_train[nn_indices]

            # Find the most common label and append it to predictions.
            predicted_label = self._majority_vote(nn_labels)
            predictions.append(predicted_label)

            if self.plot_neighbors and self.image_shape is not None:
                test_image = x.reshape(self.image_shape)
                neighbor_images = [
                    self.X_train[idx].reshape(self.image_shape) for idx in nn_indices
                ]

                save_path = None
                if self.plot_dir is not None:
                    save_path = Path(self.plot_dir) / f"{self.plot_prefix}_{sample_index:02d}.png"

                plot_knn_neighbors(
                    test_image=test_image,
                    neighbor_images=neighbor_images,
                    neighbor_labels=nn_labels,
                    test_label=None,
                    pred_label=predicted_label,
                    save_path=save_path,
                )

        # Return predictions as a NumPy array.
        return np.asarray(predictions)