import numpy as np
import matplotlib.pyplot as plt
from skimage import io, restoration, color
import argparse
from sklearn.cluster import KMeans
import cv2
from skimage.measure import regionprops
from scipy.ndimage import label
# Fix random seed for reproducibility.
np.random.seed(42)


def create_rgb_mask(binary_mask):
    # Create an all-white RGB image of the same dimensions as the binary mask
    height, width = binary_mask.shape
    rgb_mask = (
        np.ones((height, width, 3), dtype=np.uint8) * 255
    )  # Multiply by 255 to make it white

    # Where the binary mask is 1, set the RGB mask to black
    rgb_mask[binary_mask == 1] = [0, 0, 0]  # Set to black

    return rgb_mask


def main(image):
    # Denoise image using bilateral filter.
    denoised_image = restoration.denoise_bilateral(
        image, win_size=5, sigma_color=1, sigma_spatial=15, channel_axis=-1
    )

    # Use LAB colour space for clustering.
    image_lab = color.rgb2lab(denoised_image)
    pixels = image_lab.reshape((-1, 3))

    # Create the KMeans model.
    kmeans = KMeans(n_clusters=5)

    # Fit the KMeans model.
    kmeans.fit(pixels)

    # Get the cluster assignments.
    cluster_assignments = kmeans.labels_

    # Get the cluster centers.
    cluster_centers = kmeans.cluster_centers_

    # Reshape the cluster assignments to an image.
    cluster_image = cluster_assignments.reshape(image_lab.shape[:2])

    # LAB value for purple (source: epaint.co.uk https://tinyurl.com/2p9rrdfc).
    purple = [34.64, 34.45, -4.07]

    # Find the cluster center that is closest to purple.
    closest_cluster_index = np.argmin(np.linalg.norm(cluster_centers - purple, axis=1))

    # Get the mask for the purple cluster.
    mask = cluster_image == closest_cluster_index

    # Perform opening with a ball of radius 2 to disconnect small regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Keep only connected components with area greater than 100 pixels.
    label_image, _ = label(mask)
    regions = regionprops(label_image)
    mask = np.zeros_like(mask)
    for region in regions:
        if region.area > 100:
            mask[label_image == region.label] = 1

    # Visualise the mask over the original image.
    rgb_mask = create_rgb_mask(mask)
    rgb_mask = rgb_mask.astype(float)
    rgb_mask[rgb_mask == 0] = np.nan  # Set black to transparent.

    # Normalise the mask to avoid clipping when overlaying.
    rgb_mask /= 255.0

    plt.imshow(image)
    plt.imshow(rgb_mask, cmap="jet", alpha=0.5)
    plt.title("Noisy Flowers Segmentation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/q1b_segmentation.png", dpi=300, bbox_inches="tight")
    print("[INFO] Plot saved at 'outputs/q1b_segmentation.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentation algorithm for flower image."
    )
    parser.add_argument("image_path", type=str, help="Input file")
    args = parser.parse_args()

    image = io.imread(args.image_path)

    # RGBA -> RGB since alpha=255 everywhere.
    image = image[:, :, :3]
    main(image)
