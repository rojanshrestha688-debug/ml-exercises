import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover - depends on local environment
    LogisticRegression = None


class LogisticRegressionClassifier:
    def __init__(self, max_iter=2000, random_state=0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        """
        Train a logistic-regression classifier on the given feature matrix.

        Requirements:
            - convert X and y to NumPy arrays
            - validate that X has shape (n_samples, n_features)
            - validate that y is one-dimensional
            - validate that X and y contain the same number of samples
            - create and fit sklearn.linear_model.LogisticRegression
            - return self
        """
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for LogisticRegressionClassifier.")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - raise an error if fit() was not called first
            - accept either a single sample or a full batch
            - validate the feature dimension
            - return the model predictions as a NumPy array
        """
        if self.model is None:
            raise ValueError("This LogisticRegressionClassifier instance is not fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the classifier was fitted with "
                f"{self.n_features_} features."
            )

        return np.asarray(self.model.predict(X))