from pathlib import Path

from histogram_equalization import (
    compute_cdf,
    compute_histogram,
    equalize_image,
    load_image as load_image_hist,
    save_image as save_image_hist,
    show_images,
)
from noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_salt_and_pepper_noise,
    add_uniform_noise,
    load_image as load_image_noise,
    save_image as save_image_noise,
)
from otsu import load_image as load_image_otsu, otsu_binarize
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    otsu_input = data_dir / "runes.png"
    otsu_image = load_image_otsu(str(otsu_input))
    otsu_result, otsu_threshold = otsu_binarize(otsu_image)
    if otsu_result.size == 0:
        print("Otsu binarization not implemented yet.")
    else:
        show_images(otsu_image, otsu_result)
        print(f"Otsu threshold: {otsu_threshold}")

    hist_input = data_dir / "hello.png"
    hist_output = data_dir / "kitty.png"
    hist_image = load_image_hist(str(hist_input))
    histogram = compute_histogram(hist_image)
    cdf = compute_cdf(histogram)
    equalized = equalize_image(hist_image, cdf)
    if equalized.size == 0:
        print("Histogram equalization not implemented yet.")
    else:
        save_image_hist(equalized, str(hist_output))
        show_images(hist_image, equalized)

    noise_input = data_dir / "hello.png"
    noise_image = load_image_noise(str(noise_input))
    noise_tasks = [
        (add_gaussian_noise, "hello_gaussian.png"),
        (add_salt_and_pepper_noise, "hello_salt_pepper.png"),
        (add_poisson_noise, "hello_poisson.png"),
        (add_uniform_noise, "hello_uniform.png"),
    ]

    preview_noise = noise_tasks[0][0](noise_image)
    if preview_noise.size == 0:
        print("Noise generation not implemented yet.")
    else:
        save_image_noise(preview_noise, str(data_dir / noise_tasks[0][1]))
        for generator, output_name in noise_tasks[1:]:
            noisy = generator(noise_image)
            if noisy.size == 0:
                print("Noise generation not fully implemented yet.")
                break
            save_image_noise(noisy, str(data_dir / output_name))
