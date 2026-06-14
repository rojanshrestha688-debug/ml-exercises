import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    # The function receives a padded image (pad_image) and the current pixel of our padded image.
    # ToDo: Return the surrounding area around that center pixel with the given size (window_size).
    # ToDo: Use slicing.
    half = window_size // 2  # how far the window reaches on each side of the centre
    return padded_image[center_row - half:center_row + half + 1,
                        center_col - half:center_col + half + 1]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    # Pad the image with zeros.
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply erosion on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    half = se_size // 2
    padded_image = pad_image(image, half)  # zero border so the SE never falls off the edge
    output = np.zeros_like(image)

    # `active` marks the SE cells we actually test. Using it (instead of comparing
    # the whole window to the SE) makes erosion correct for ANY SE shape.
    active = structuring_element == 1

    # ToDo: Iterate over the provided image and perform erosion around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # centre in the padded image is shifted by `half`
            region = extract_region(padded_image, i + half, j + half, se_size)
            # keep the pixel only if ALL active SE positions sit on foreground
            if np.all(region[active] == 1):
                output[i, j] = 1
    return output


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply dilation on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    half = se_size // 2
    padded_image = pad_image(image, half)
    output = np.zeros_like(image)

    active = structuring_element == 1

    # ToDo: Iterate over the provided image and perform dilation around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = extract_region(padded_image, i + half, j + half, se_size)
            # set the pixel if AT LEAST ONE active SE position sees foreground
            if np.any(region[active] == 1):
                output[i, j] = 1
    return output


def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform opening (erosion followed by dilation).
    result = input_image.copy()
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)
        result = dilate_binary(result, structuring_element)
    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform closing (dilation followed by erosion).
    result = input_image.copy()
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)
        result = erode_binary(result, structuring_element)
    return result


# ---------------------------------------------------------------------------
# Small, dependency-free helpers used ONLY to decide when to stop iterating.
# They use a simple flood fill (a stack of pixels to visit) so we do not need
# any extra library. 8-connectivity is used for the foreground.
# ---------------------------------------------------------------------------
def _flood_fill(mask: np.ndarray, seeds) -> np.ndarray:
    # Return a boolean array of all cells reachable from `seeds`, moving only
    # through True cells of `mask` (8-connected neighbourhood).
    visited = np.zeros_like(mask, dtype=bool)
    stack = [s for s in seeds if mask[s]]
    for s in stack:
        visited[s] = True
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return visited


def count_foreground_blobs(image: np.ndarray) -> int:
    # Count how many separate foreground (==1) regions exist.
    fg = image == 1
    seen = np.zeros_like(fg, dtype=bool)
    blobs = 0
    for r in range(fg.shape[0]):
        for c in range(fg.shape[1]):
            if fg[r, c] and not seen[r, c]:
                seen |= _flood_fill(fg, [(r, c)])  # mark this whole blob as seen
                blobs += 1
    return blobs


def has_enclosed_hole(image: np.ndarray) -> bool:
    # A "hole" is a background (==0) region NOT connected to the image border.
    # We flood the background from all border pixels; any background pixel left
    # unreached must be an enclosed hole.
    bg = image == 0
    border_seeds = (
        [(0, c) for c in range(bg.shape[1])] +
        [(bg.shape[0] - 1, c) for c in range(bg.shape[1])] +
        [(r, 0) for r in range(bg.shape[0])] +
        [(r, bg.shape[1] - 1) for r in range(bg.shape[0])]
    )
    reachable = _flood_fill(bg, border_seeds)
    return bool(np.any(bg & ~reachable))


def load_binary(filepath: str) -> np.ndarray:
    # Load the image and binarize it again with a simple threshold.
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)  # type: ignore
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    # Save the binary image.
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Paths.
    raw_erosion_image_path = 'data/erosion_image_raw.png'
    raw_dilation_image_path = 'data/dilation_image_raw.png'
    erosion_out_path = 'data/erosion_output.png'
    dilation_out_path = 'data/dilation_output.png'

    # Load images.
    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    # Structuring element.
    SE = np.ones((5, 5), dtype=np.uint8)

    # Erosion.
    # ToDo: Perform erosion multiple times until the circles separate from each other.
    # Stop as soon as the foreground splits into 2 separate blobs.
    eroded = erosion_input.copy()
    erosion_iterations = 0
    while count_foreground_blobs(eroded) < 2 and eroded.sum() > 0:
        eroded = erode_binary(eroded, SE)
        erosion_iterations += 1
    print(f"Erosion: circles separated after {erosion_iterations} iteration(s) with a {SE.shape[0]}x{SE.shape[1]} SE.")
    save_binary(eroded, erosion_out_path)
    show_image(eroded, f"Erosion Output ({erosion_iterations} iterations)")

    # Dilation.
    # ToDo: Perform dilation multiple times until the hole closes.
    # Stop as soon as there is no enclosed background region left.
    dilated = dilation_input.copy()
    dilation_iterations = 0
    while has_enclosed_hole(dilated):
        dilated = dilate_binary(dilated, SE)
        dilation_iterations += 1
    print(f"Dilation: hole filled after {dilation_iterations} iteration(s) with a {SE.shape[0]}x{SE.shape[1]} SE.")
    save_binary(dilated, dilation_out_path)
    show_image(dilated, f"Dilation Output ({dilation_iterations} iterations)")