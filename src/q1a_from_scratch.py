from from_scratch_seg_funcs import (
    otsu_threshold_from_scratch,
    label_from_scratch,
    binary_fill_holes_from_scratch,
)
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2  # Only used for reading the image.


def main(image):
    # Visualise original image.
    plt.imshow(image, cmap="gray")
    plt.title("Original image")
    plt.show()

    # Apply an Otus threshold.
    thresh = otsu_threshold_from_scratch(image)
    binary = image > thresh
    plt.imshow(binary, cmap="gray")
    plt.title("Binary image")
    plt.show()

    # Label connected components.
    labelled_image = label_from_scratch(~binary, connectivity=2)
    plt.imshow(labelled_image, cmap="tab20")
    plt.title("Connected components")
    plt.show()

    # Select the top 3rd and 4th largest connected components.
    # (Assuming background and body are the top 2 largest components...)
    component_sizes = np.bincount(labelled_image.flat)
    largest_components = np.argsort(component_sizes)[::-1][2:4]

    # Create a mask for the lungs.
    mask = np.zeros_like(labelled_image)
    mask[np.isin(labelled_image, largest_components)] = 1
    plt.imshow(mask, cmap="gray")
    plt.title("Lung mask")
    plt.show()

    # Fill holes in the mask.
    mask = binary_fill_holes_from_scratch(mask)
    plt.imshow(mask, cmap="gray")
    plt.title("Filled lung mask")
    plt.show()

    # Apply mask to original image.
    mask = mask.astype(float)
    mask[mask == 0] = np.nan
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, cmap="coolwarm", alpha=0.5)
    plt.title("Final Lung CT Segmentation (from scratch)")
    plt.axis("off")
    plt.savefig("output_segs/q1a_from_scratch.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentation algorithm for lung CT image."
    )
    parser.add_argument("image_path", type=str, help="Input file")
    args = parser.parse_args()

    image_path = args.image_path

    # Read the image.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Segment the image.
    main(image)
