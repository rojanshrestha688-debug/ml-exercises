import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    # Load the image using CV2 and return it.
    loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return loaded_image


def compute_histogram(image: np.ndarray) -> np.ndarray:
    # ToDo: Create a histogram for the given image (256 values).
    # ToDo: Don't use functions like np.histogram.
    # ToDo: It is easier if you flatten your image first.
    histogram = np.zeros(256)
    flat = image.flatten() #flattens the image into a 1D array
    for pixel in flat:
        histogram[pixel] += 1  #puts the pixel value in the corresponding bin
    return histogram


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    # ToDo: Compute the CDF.
    # ToDo: Don't forget to normalize it (turn it into a distribution).
    pdf = histogram / histogram.sum() #normalized the histogram
    cdf = np.cumsum(pdf) #cumulative distribution function
    return cdf


def equalize_image(image: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    # ToDo: Apply histogram equalization to the given image.
    # ToDo: Hint: Flatten the image first and reshape it again in the end.
    flat = image.flatten()
    cdf_min = cdf[cdf > 0].min() #find the minimum value in the cdf greater than 0
    equalized_image = np.floor(255 * ((cdf[flat] - cdf_min) / ( 1 - cdf_min )))
    equalized_image = np.clip(equalized_image, 0, 255)
    equalized_image = equalized_image.astype(np.uint8)
    equalized_image = equalized_image.reshape(image.shape) # reshape the image to the original shape
    #print(image.shape) image.shape gives you the original dimensions of your image
    return equalized_image


def save_image(image: np.ndarray, path: str) -> None:
    # Save the image to the given folder.
    cv2.imwrite(path, image)


def show_images(original_image: np.ndarray, equalized_image: np.ndarray) -> None:
    # ToDo: Display the original and the equalized images next to each other.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def histogram_equalization(input_path: str, output_path: str) -> None:
    # ToDo: Combine the different functions into one.
    loaded_image = load_image(input_path)
    histogram = compute_histogram(loaded_image)
    cdf = compute_cdf(histogram)
    equalized_image = equalize_image(loaded_image, cdf)
    if equalized_image.size != 0:
        save_image(equalized_image, output_path)


if __name__ == '__main__':
    # Load the images and perform histogram equalization.
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    input_image_path = str(data_dir / 'hello.png')
    output_image_path = str(data_dir / 'kitty.png')
    histogram_equalization(input_image_path, output_image_path)

    # Show the images next to each other.
    original = load_image(input_image_path)
    if Path(output_image_path).exists():
        equalized = load_image(output_image_path)
        show_images(original, equalized)
