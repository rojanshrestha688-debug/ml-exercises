from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _finish_figure(fig, save_path=None, show=False):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show and save_path is None:
        plt.show()

    plt.close(fig)


def plot_knn_neighbors(
    test_image,
    neighbor_images,
    neighbor_labels,
    test_label=None,
    pred_label=None,
    figsize=(12, 3),
    save_path=None,
    show=False,
):
    """
    Visualize one test image together with its k nearest neighbours.

    Parameters:
        test_image (np.ndarray): Test image with shape (H, W) or (H, W, C).
        neighbor_images (list[np.ndarray]): Images of the nearest neighbours.
        neighbor_labels (list | np.ndarray): Labels of the nearest neighbours.
        test_label: Optional true label of the test image.
        pred_label: Optional predicted label of the test image.
        figsize (tuple): Figure size.
        save_path (str | Path | None): Save figure to this file if given.
        show (bool): Display the figure interactively when no save path is used.
    """
    k = len(neighbor_images)
    fig, axes = plt.subplots(1, k + 1, figsize=figsize)

    if k == 0:
        axes = [axes]

    axes[0].imshow(test_image.squeeze(), cmap="gray" if test_image.ndim == 2 else None)
    title = "Test"
    if test_label is not None:
        title += f"\nTrue: {test_label}"
    if pred_label is not None:
        title += f"\nPred: {pred_label}"
    axes[0].set_title(title)
    axes[0].axis("off")

    for index, (img, label) in enumerate(zip(neighbor_images, neighbor_labels), start=1):
        axes[index].imshow(img.squeeze(), cmap="gray" if img.ndim == 2 else None)
        axes[index].set_title(f"Neighbor {index}\nLabel: {label}")
        axes[index].axis("off")

    fig.tight_layout()
    _finish_figure(fig, save_path=save_path, show=show)


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None, show=False):
    """
    Plot a confusion matrix.

    Parameters:
        cm (np.ndarray): Confusion matrix with shape (n_classes, n_classes).
        class_names (list[str]): Display names for the classes.
        title (str): Figure title.
        save_path (str | Path | None): Save figure to this file if given.
        show (bool): Display the figure interactively when no save path is used.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("Predicted", labelpad=20)
    ax.set_ylabel("True", labelpad=20)
    ax.set_title(title, pad=40)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center", color="black", fontsize=12)

    fig.tight_layout()
    _finish_figure(fig, save_path=save_path, show=show)


def plot_accuracy_comparison(acc_dict, title="Classifier Accuracy Comparison", save_path=None, show=False):
    """
    Plot a bar chart comparing classifier accuracies.

    Parameters:
        acc_dict (dict[str, float]): Mapping from classifier name to accuracy.
        title (str): Figure title.
        save_path (str | Path | None): Save figure to this file if given.
        show (bool): Display the figure interactively when no save path is used.
    """
    names = list(acc_dict.keys())
    accuracies = list(acc_dict.values())

    palette = ["skyblue", "salmon", "lightgreen", "gold", "plum", "lightslategray"]
    colors = [palette[index % len(palette)] for index in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4))
    ax.bar(names, accuracies, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)

    for index, accuracy in enumerate(accuracies):
        ax.text(index, accuracy + 0.02, f"{accuracy:.2f}", ha="center")

    fig.tight_layout()
    _finish_figure(fig, save_path=save_path, show=show)
