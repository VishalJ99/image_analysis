import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from scipy.ndimage import label
import argparse


def main(image):
    # Apply an Otus threshold.
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep 2nd and 3rd largest connected components.
    label_image, _ = label(~binary)
    regions = regionprops(label_image)
    regions.sort(key=lambda x: x.area, reverse=True)
    binary = np.zeros_like(binary)

    # Avoid largest connected component (background).
    binary[label_image == regions[1].label] = 1
    binary[label_image == regions[2].label] = 1

    # Fill holes.
    binary = binary_fill_holes(binary)

    # Visualise the final segmentation over the original image.
    plt.imshow(image, cmap="gray")

    # Only show the segmented regions.
    binary = binary.astype(float)
    binary[binary == 0] = np.nan
    plt.imshow(binary, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.title("Lung CT Segmentation")
    plt.savefig("outputs/q1a_segmentation.png")
    plt.tight_layout()

    print("[INFO] Plot saved at 'outputs/q1a_segmentation.png'")


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
