from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from DistanceMeasure import euclideanDistance, mseDistance
from HOGFeature import calculateHOG
from kaktovikAlignmentSimple import simpleAlignment


BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "img"
RETRIEVAL_DIR = IMG_DIR / "retrieval"


def load_grayscale(name):
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


def print_comparison(name, img_a, img_b):
    hog_a = calculateHOG(img_a)
    hog_b = calculateHOG(img_b)
    print(name)
    print(f"  MSE: {mseDistance(img_a, img_b):.2f}")
    print(f"  HOG distance: {euclideanDistance(hog_a, hog_b):.2f}")


def prepare_image(img, align):
    if align:
        return simpleAlignment(img)
    return cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)


def top_k_accuracy(items, distance_fn, k):
    correct = 0
    for i, (label, _, _) in enumerate(items):
        ranked = []
        for j, (other_label, other_name, _) in enumerate(items):
            if i == j:
                continue
            ranked.append((distance_fn(i, j), other_label, other_name))
        ranked.sort(key=lambda row: row[0])
        predicted_labels = [row[1] for row in ranked[:k]]
        correct += label in predicted_labels
    return correct / len(items)


def top_matches(items, distance_fn, query_name, k):
    query_index = None
    for i, (_, name, _) in enumerate(items):
        if name == query_name:
            query_index = i
            break

    if query_index is None:
        raise ValueError(f"Unknown query image: {query_name}")

    ranked = []
    for j, (label, name, _) in enumerate(items):
        if query_index == j:
            continue
        ranked.append((distance_fn(query_index, j), label, name))
    ranked.sort(key=lambda row: row[0])
    return ranked[:k]


def prepare_gallery(items, use_alignment):
    prepared_images = [prepare_image(img, use_alignment) for _, _, img in items]
    hog_features = [calculateHOG(img) for img in prepared_images]
    return prepared_images, hog_features


def rank_query_against_gallery(query_img, prepared_images, hog_features, items, use_alignment, metric):
    prepared_query = prepare_image(query_img, use_alignment)
    ranked = []

    if metric == "hog":
        query_feature = calculateHOG(prepared_query)
        for j, (label, name, _) in enumerate(items):
            ranked.append((euclideanDistance(query_feature, hog_features[j]), label, name))
    elif metric == "mse":
        for j, (label, name, _) in enumerate(items):
            ranked.append((mseDistance(prepared_query, prepared_images[j]), label, name))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    ranked.sort(key=lambda row: row[0])
    return ranked


def print_retrieval_results(name, items):
    prepared_images, hog_features = prepare_gallery(items, use_alignment=True)

    hog_top1 = top_k_accuracy(items, lambda i, j: euclideanDistance(hog_features[i], hog_features[j]), 1)
    hog_top3 = top_k_accuracy(items, lambda i, j: euclideanDistance(hog_features[i], hog_features[j]), 3)
    hog_top5 = top_k_accuracy(items, lambda i, j: euclideanDistance(hog_features[i], hog_features[j]), 5)

    mse_top1 = top_k_accuracy(items, lambda i, j: mseDistance(prepared_images[i], prepared_images[j]), 1)
    mse_top3 = top_k_accuracy(items, lambda i, j: mseDistance(prepared_images[i], prepared_images[j]), 3)
    mse_top5 = top_k_accuracy(items, lambda i, j: mseDistance(prepared_images[i], prepared_images[j]), 5)

    print(name)
    print(f"  HOG top-1/top-3/top-5: {hog_top1:.2f} / {hog_top3:.2f} / {hog_top5:.2f}")
    print(f"  MSE top-1/top-3/top-5: {mse_top1:.2f} / {mse_top3:.2f} / {mse_top5:.2f}")

    query_name = "class10_kaktovik-000_rot_0.jpg"
    print(f"  Example HOG retrieval for query {query_name}:")
    for distance, label, match_name in top_matches(
        items,
        lambda i, j: euclideanDistance(hog_features[i], hog_features[j]),
        query_name,
        3,
    ):
        print(f"    {match_name} ({label}) -> {distance:.2f}")


def print_transformed_query_retrieval(items, query_name):
    query_item = next((item for item in items if item[1] == query_name), None)
    if query_item is None:
        raise ValueError(f"Unknown query image: {query_name}")

    label, _, original = query_item
    translated = translate_image(original, 18, -10)
    rotated = rotate_image(original, 18)

    prepared_images, hog_features = prepare_gallery(items, use_alignment=True)

    print("\nRetrieval of transformed queries against the real gallery after alignment (HOG):")
    for variant_name, variant_img in (("translated", translated), ("rotated", rotated)):
        ranked = rank_query_against_gallery(
            variant_img,
            prepared_images,
            hog_features,
            items,
            use_alignment=True,
            metric="hog",
        )
        top3 = ranked[:3]
        top3_labels = [entry[1] for entry in top3]
        print(f"  {variant_name} query, correct label in top-3: {label in top3_labels}")
        for distance, match_label, match_name in top3:
            print(f"    {match_name} ({match_label}) -> {distance:.2f}")


img1 = load_grayscale("kaktovik-008_rot_0.jpg")
img2 = load_grayscale("kaktovik-012_rot_0.jpg")
img3 = load_grayscale("kaktovik-012_rot_2.jpg")
img4 = load_grayscale("kaktovik-001_rot_0.jpg")

aligned1 = simpleAlignment(img1)
aligned2 = simpleAlignment(img2)
aligned3 = simpleAlignment(img3)
aligned4 = simpleAlignment(img4)

print("Comparison results on aligned symbols:")
print_comparison("1 vs 1", aligned1, aligned1)
print_comparison("same symbol, different writer", aligned1, aligned2)
print_comparison("same symbol, different rotation", aligned2, aligned3)
print_comparison("different symbols", aligned2, aligned4)

translated = translate_image(img2, 18, -10)
rotated = rotate_image(img2, 18)

print("\nRobustness experiment:")
print_comparison("Translated, after alignment", aligned2, simpleAlignment(translated))
print_comparison("Rotated, after alignment", aligned2, simpleAlignment(rotated))

retrieval_items = load_retrieval_items()
class_counts = Counter(label for label, _, _ in retrieval_items)
print("\nRetrieval experiment on the provided gallery:")
print(
    f"Gallery size: {len(retrieval_items)} images from {len(class_counts)} classes "
    f"with {min(class_counts.values())} to {max(class_counts.values())} examples per class."
)
print_retrieval_results("Retrieval on aligned gallery images", retrieval_items)
print_transformed_query_retrieval(retrieval_items, "class10_kaktovik-000_rot_0.jpg")

fig, axs = plt.subplots(2, 4, figsize=(12, 6))
axs[0, 0].imshow(img1, cmap="gray"); axs[0, 0].set_title("Input 1"); axs[0, 0].axis("off")
axs[0, 1].imshow(img2, cmap="gray"); axs[0, 1].set_title("Input 2"); axs[0, 1].axis("off")
axs[0, 2].imshow(translated, cmap="gray"); axs[0, 2].set_title("Translated"); axs[0, 2].axis("off")
axs[0, 3].imshow(rotated, cmap="gray"); axs[0, 3].set_title("Rotated"); axs[0, 3].axis("off")

axs[1, 0].imshow(aligned1, cmap="gray"); axs[1, 0].set_title("Aligned 1"); axs[1, 0].axis("off")
axs[1, 1].imshow(aligned2, cmap="gray"); axs[1, 1].set_title("Aligned 2"); axs[1, 1].axis("off")
axs[1, 2].imshow(simpleAlignment(translated), cmap="gray"); axs[1, 2].set_title("Aligned translated"); axs[1, 2].axis("off")
axs[1, 3].imshow(simpleAlignment(rotated), cmap="gray"); axs[1, 3].set_title("Aligned rotated"); axs[1, 3].axis("off")

plt.tight_layout()
plt.show()
