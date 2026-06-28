from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    from .Eigenfaces import (
        classify_image,
        create_database_from_folder,
        reconstruct_image,
        train_both_classifiers,
    )
except ImportError:
    from Eigenfaces import (
        classify_image,
        create_database_from_folder,
        reconstruct_image,
        train_both_classifiers,
    )


BASE_DIR = Path(__file__).resolve().parent
FACE_DATASET_DIR = BASE_DIR / "eigenfaces"
KAKTOVIK_DATASET_DIR = BASE_DIR / "kaktovik"


def evaluate_dataset(
    dataset_name,
    dataset_dir,
    image_size=(64, 64),
    reconstruction_components=(5, 20, 50),
    num_eigenfaces=40,
):
    labels, data, num_images, h, w = create_database_from_folder(dataset_dir, image_size=image_size)
    print(f"\n{dataset_name}")
    print(f"  Images: {num_images}")
    print(f"  Classes: {len(np.unique(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.25,
        random_state=0,
        stratify=labels,
    )

    num_eigenfaces = min(num_eigenfaces, X_train.shape[0] - 1, X_train.shape[1])
    eigenfaces, num_eigenfaces, avg = train_both_classifiers(
        y_train,
        X_train,
        X_train.shape[0],
        h,
        w,
        num_eigenfaces=num_eigenfaces,
    )

    for classifier_type in ("logistic", "gaussian_nb"):
        predictions = np.empty(X_test.shape[0], dtype=object)
        for index, image in enumerate(X_test):
            predictions[index] = classify_image(
                image,
                eigenfaces,
                avg,
                num_eigenfaces,
                h,
                w,
                classifier_type=classifier_type,
            )[0]

        print(f"\n  {classifier_type}")
        print(classification_report(y_test, predictions, zero_division=0))

    sample = X_test[0]
    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{dataset_name}: PCA reconstruction")
    plt.subplot(1, len(reconstruction_components) + 1, 1)
    plt.imshow(sample.reshape(h, w), cmap="gray")
    plt.title("Input")
    plt.axis("off")

    for plot_index, num_components in enumerate(reconstruction_components, start=2):
        clipped_components = min(num_components, num_eigenfaces)
        reconstruction = reconstruct_image(sample, eigenfaces, avg, clipped_components, h, w)
        plt.subplot(1, len(reconstruction_components) + 1, plot_index)
        plt.imshow(reconstruction, cmap="gray")
        plt.title(f"{clipped_components} PCs")
        plt.axis("off")

    plt.tight_layout()


def main():
    datasets = [
        ("Faces", FACE_DATASET_DIR, (64, 64), (5, 20, 50), 50),
        ("Kaktovik", KAKTOVIK_DATASET_DIR, (64, 64), (5, 20, 30), 20),
    ]

    for dataset_name, dataset_dir, image_size, reconstruction_components, num_eigenfaces in datasets:
        if dataset_dir.exists():
            evaluate_dataset(
                dataset_name,
                dataset_dir,
                image_size=image_size,
                reconstruction_components=reconstruction_components,
                num_eigenfaces=num_eigenfaces,
            )
        else:
            print(f"Skipping {dataset_name}: missing dataset directory {dataset_dir}")

    plt.show()


if __name__ == "__main__":
    main()
