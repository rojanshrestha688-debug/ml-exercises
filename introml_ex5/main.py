from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from evaluation import confusion_matrix
from knn import KNNClassifier
from logistic_regression import LogisticRegressionClassifier
from nbnn import NBNNClassifier
from visualization import plot_accuracy_comparison, plot_confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "images"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent / "introml_ex5" / "images"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UNIT_TEST_DIR = DATA_DIR / "unit_test"
OUTPUT_DIR = BASE_DIR / "outputs"
IMG_SIZE = (64, 64)


def load_images_labels(folder, image_size=IMG_SIZE):
    """
    Load all images from a class-folder structure and return flattened arrays.

    Parameters:
        folder (str | Path): Directory containing subfolders such as CLASS_0.
        image_size (tuple[int, int]): Target image size (width, height).

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix and label array.
    """
    images = []
    labels = []
    folder = Path(folder)

    for class_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("L")
                image = image.resize(image_size)
                image_array = np.asarray(image, dtype=np.float64) / 255.0

            images.append(image_array.reshape(-1))
            labels.append(class_dir.name)

    return np.asarray(images), np.asarray(labels)


def encode_labels(*label_arrays):
    """
    Encode one or more label arrays into consecutive integers.

    Returns:
        classes (np.ndarray): Sorted class names.
        class_to_int (dict): Mapping from class name to integer index.
        encoded_arrays (list[np.ndarray]): Encoded label arrays in input order.
    """
    classes = np.unique(np.concatenate(label_arrays))
    class_to_int = {label: index for index, label in enumerate(classes)}
    encoded_arrays = [
        np.asarray([class_to_int[label] for label in labels], dtype=int)
        for labels in label_arrays
    ]
    return classes, class_to_int, encoded_arrays


def save_confusion_plot(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, title=title, save_path=save_path)


def main(k_test_samples=None, random_state=0):
    """
    Run the Exercise 5 evaluation pipeline on the Kaktovik dataset.

    The script generates:
        - 11 neighbour plots on the unit-test set
        - confusion matrix plots for all classifier variants
        - one accuracy comparison plot

    By default, the quantitative evaluation uses the full official test split.
    The optional argument k_test_samples exists only for debugging.
    """
    print("Loading training data...")
    X_train, y_train = load_images_labels(TRAIN_DIR)
    print("Loading test data...")
    X_test, y_test = load_images_labels(TEST_DIR)
    print("Loading unit-test data...")
    X_unit, y_unit = load_images_labels(UNIT_TEST_DIR)
    print(f"Dataset sizes: train={len(X_train)}, test={len(X_test)}, unit_test={len(X_unit)}")

    if k_test_samples is not None and k_test_samples < len(X_test):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_test), size=k_test_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"Debug mode: evaluating on a reduced subset of {len(X_test)} test images.")
    else:
        print("Evaluating on the full test split.")

    classes, _, encoded = encode_labels(y_train, y_test, y_unit)
    y_train_enc, y_test_enc, y_unit_enc = encoded
    class_names = [label.replace("CLASS_", "") for label in classes]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating 11 KNN neighbour plots from the unit-test set...")
    knn_plotter = KNNClassifier(
        n_neighbors=3,
        metric="euclidean",
        plot_neighbors=True,
        image_shape=IMG_SIZE,
        plot_dir=OUTPUT_DIR / "knn_neighbors",
        plot_prefix="knn_class",
    )
    knn_plotter.fit(X_train, y_train_enc)
    unit_predictions = knn_plotter.predict(X_unit)

    predictions = {}
    accuracies = {}

    for classifier_name, classifier in (
        ("KNN (euclidean)", KNNClassifier(n_neighbors=3, metric="euclidean", image_shape=IMG_SIZE)),
        ("KNN (cosine)", KNNClassifier(n_neighbors=3, metric="cosine", image_shape=IMG_SIZE)),
        ("NBNN (euclidean)", NBNNClassifier(metric="euclidean")),
        ("NBNN (cosine)", NBNNClassifier(metric="cosine")),
    ):
        print(f"Running {classifier_name}...")
        classifier.fit(X_train, y_train_enc)
        y_pred = classifier.predict(X_test)
        predictions[classifier_name] = y_pred
        accuracies[classifier_name] = float(np.mean(y_test_enc == y_pred))
        print(f"Accuracy ({classifier_name}): {accuracies[classifier_name]:.4f}")

    print("Running Logistic Regression...")
    logreg = LogisticRegressionClassifier(max_iter=2000, random_state=random_state)
    logreg.fit(X_train, y_train_enc)
    y_pred_logreg = logreg.predict(X_test)
    predictions["Logistic Regression"] = y_pred_logreg
    accuracies["Logistic Regression"] = float(np.mean(y_test_enc == y_pred_logreg))
    print(f"Accuracy (logistic): {accuracies['Logistic Regression']:.4f}")

    for name, y_pred in predictions.items():
        save_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_confusion_plot(
            y_test_enc,
            y_pred,
            class_names,
            title=f"Confusion Matrix - {name}",
            save_path=OUTPUT_DIR / f"confusion_{save_name}.png",
        )

    plot_accuracy_comparison(
        accuracies,
        save_path=OUTPUT_DIR / "accuracy_comparison.png",
    )

    unit_accuracy = float(np.mean(unit_predictions == y_unit_enc))
    print(f"Unit-test accuracy of KNN (euclidean): {unit_accuracy:.4f}")
    print(f"Saved plots to: {OUTPUT_DIR}")
    return accuracies


if __name__ == "__main__":
    main()
