import argparse
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from copy import deepcopy


def main(image_path, level, crop_list):
    # Read the image.
    img = io.imread("data/river_side.jpeg")

    # Convert RGB to gray scale.
    img_gray = np.mean(img, axis=2)
    level = 2

    img_gray = img_gray[crop_list[0]: crop_list[1], crop_list[2]: crop_list[3]]

    coeffs = pywt.wavedec2(img_gray, wavelet="db4", mode="per", level=level)
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)

    # Threshold wavelet coefficients retaining only the largest 15%.
    coeff_array_top_15_binary = np.abs(coeff_array) > np.percentile(
        np.abs(coeff_array), 85
    )

    plt.imshow(coeff_array_top_15_binary, cmap="gray")
    plt.axis("off")
    plt.title("Thresholded Wavelet Coefficients - Top 15%")
    plt.tight_layout()
    plt.savefig("outputs/q2c_thresholded_wavelet_coeffs.png", dpi=300)
    print(
        "[INFO] Thresholded coefficient plot saved to "
        "'outputs/q2c_thresholded_wavelet_coeffs.png'"
    )

    # Reconstruct the image using the thresholded coefficients.
    coeffs = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format="wavedec2")

    reconstructed_img = pywt.waverec2(coeffs, wavelet="db4", mode="per")

    plt.imshow(reconstructed_img, cmap="gray")
    plt.axis("off")
    plt.title("Reconstructed Image")
    plt.tight_layout()
    plt.savefig("outputs/q2c_reconstructed_image.png", dpi=300)
    print("[INFO] Reconstructed image saved to 'outputs/q2c_reconstructed_image.png'")

    # Compute diff between original and reconstructed image.
    diff = img_gray - reconstructed_img
    plt.imshow(diff, cmap="gray")
    plt.axis("off")
    plt.colorbar()
    plt.title(f"Difference Image (MSE: {np.mean(diff ** 2):.2f})")
    plt.tight_layout()
    plt.savefig("outputs/q2c_diff_image.png", dpi=300)
    print("[INFO] Difference image saved to 'outputs/q2c_diff_image.png'")

    # Visualise reconstruction when retaining diff percentages of top coeffs.
    thresholds = [0.2, 0.1, 0.05, 0.025]
    for thresh in thresholds:
        coeff_array_dup = deepcopy(coeff_array)
        coeff_threshold = np.percentile(np.abs(coeff_array), 100 * (1 - thresh))

        # Zeroing out smaller coefficients
        coeff_array_dup[np.abs(coeff_array) < coeff_threshold] = 0

        # Convert array back to original coefficient structure
        coeffs_thresh = pywt.array_to_coeffs(
            coeff_array_dup, coeff_slices, output_format="wavedec2"
        )

        # Reconstruct image from thresholded coefficients.
        img_reconstructed = pywt.waverec2(coeffs_thresh, "db4", mode="periodization")

        # Compute difference and MSE
        difference = img_gray - img_reconstructed
        mse = np.mean((difference) ** 2)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img_gray, cmap="gray", interpolation="nearest")
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(img_reconstructed, cmap="gray", interpolation="nearest")
        ax[1].set_title("Reconstructed Image")
        ax[1].axis("off")

        im = ax[2].imshow(difference, cmap="coolwarm", interpolation="nearest")
        ax[2].set_title(f"Difference Image\nMSE: {mse:.2e}")
        ax[2].axis("off")
        fig.colorbar(im, ax=ax[2], fraction=0.036, pad=0.04)

        plt.tight_layout()
        plt.savefig(
            f"outputs/q2c_river_side_threshold_{thresh}.jpeg",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"[INFO] Thresholded coefficient plot saved to "
            f"'outputs/q2c_river_side_threshold_{thresh}.jpeg'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wavelet Reconstruction of river image."
    )
    parser.add_argument("image_path", type=str, help="Input file")
    parser.add_argument("--level", type=int,
                        help="Level of wavelet decomposition")
    parser.add_argument(
        "--crop_list",
        nargs="+",
        type=int,
        help="List of crop coordinates",
        default=[100, -100, 220, -220],
    )
    args = parser.parse_args()

    image_path = args.image_path
    level = args.level
    crop_list = args.crop_list
    main(image_path, level, crop_list)
