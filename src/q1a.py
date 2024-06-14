import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from scipy.ndimage import label
import argparse


def main(image):
    # Visualise original image.
    plt.imshow(image, cmap="gray")
    plt.title("Original image")
    plt.show()

    # Apply an Otus threshold.
    _, binary = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    plt.title("Binary image")
    plt.tight_layout()

    plt.show()

    # Keep 2nd and 3rd largest connected components.
    label_image, _ = label(~binary)
    regions = regionprops(label_image)
    regions.sort(key=lambda x: x.area, reverse=True)
    binary = np.zeros_like(binary)

    # Avoid largest connected component (background).
    binary[label_image == regions[1].label] = 1
    binary[label_image == regions[2].label] = 1
    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    plt.title("Binary image with 2 largest connected components "
              "(excluding background)")
    plt.tight_layout()
    plt.show()

    # Fill holes.
    binary = binary_fill_holes(binary)
    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    plt.title("Binary image with filled holes")
    plt.tight_layout()
    plt.show()

    # Visualise the final segmentation over the original image.
    plt.imshow(image, cmap="gray")
    binary = binary.astype(float)
    binary[binary == 0] = np.nan  # Only show the segmented regions.
    plt.imshow(binary, alpha=0.5, cmap="jet")
    plt.tight_layout()
    plt.axis("off")
    plt.title("Final Lung CT Segmentation")
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
