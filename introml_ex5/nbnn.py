import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Store training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - validate shapes
            - store the sorted unique class labels in self.classes_
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
        self.classes_ = np.unique(y)
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        diff = self.X_train - x
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Use the same convention as in knn.py:
            cosine_distance = 1 - cosine_similarity
        """
        train_norms = np.linalg.norm(self.X_train, axis=1)
        x_norm = np.linalg.norm(x)

        denom = train_norms * x_norm
        denom = np.where(denom == 0, 1e-10, denom)

        dot_products = self.X_train @ x
        cosine_similarity = dot_products / denom

        return 1 - cosine_similarity

    def _class_scores(self, distances):
        """
        Compute one score per class.

        For each class, use the distance of the nearest training sample from
        that class. The predicted class is the class with the smallest score.
        """
        scores = np.array(
            [distances[self.y_train == label].min() for label in self.classes_]
        )
        return scores

    def predict(self, X):
        """
        Predict labels for one or more samples with the NBNN rule.

        Requirements:
            - allow either a single sample or a batch
            - compute distances to all training samples
            - convert them into class-wise scores
            - return the class label with the smallest score
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("This NBNNClassifier instance is not fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            if self.metric == "euclidean":
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                distances = self._cosine_distances(x)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            scores = self._class_scores(distances)
            best_class_index = np.argmin(scores)
            predictions.append(self.classes_[best_class_index])

        return np.asarray(predictions)