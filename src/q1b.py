import numpy as np
import matplotlib.pyplot as plt
from skimage import io, restoration, color
import argparse
from sklearn.cluster import KMeans
import cv2
from skimage.measure import regionprops
from scipy.ndimage import label


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
    # Visualise original image.
    plt.imshow(image)
    plt.title("RGB Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("report/figs/q1b_original.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Denoise image using bilateral filter.
    denoised_image = restoration.denoise_bilateral(
        image, win_size=5, sigma_color=1, sigma_spatial=15,
        channel_axis=-1
    )

    plt.imshow(denoised_image)
    # plt.title("Denoised Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("report/figs/q1b_denoised.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Fit a KMeans model to the image.
    kmeans = KMeans(n_clusters=4)

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

    # Visualise the mask
    plt.imshow(mask, cmap="gray")
    # plt.title("K Means Purple Cluster Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("report/figs/q1b_kmeans_mask.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Keep only connected components with area greater than 100 pixels.
    label_image, _ = label(mask)
    regions = regionprops(label_image)
    mask = np.zeros_like(mask)
    for region in regions:
        if region.area > 100:
            mask[label_image == region.label] = 1

    plt.imshow(mask, cmap="gray")
    # plt.title("Post Removing Small Connected Component Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("output_segs/q1b_mask_post_cnc_removal.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Visualise the mask over the original image.
    rgb_mask = create_rgb_mask(mask)
    rgb_mask = rgb_mask.astype(float)
    rgb_mask[rgb_mask == 0] = np.nan  # Set black to transparent.

    # Normalise the mask to avoid clipping when overlaying.
    rgb_mask /= 255.0

    plt.imshow(image)
    plt.imshow(rgb_mask, cmap="jet", alpha=0.5)  # Use a colormap with good contrast.
    # plt.title("Mask Overlay")
    plt.axis("off")
    plt.savefig("output_segs/q1b_final_mask.png", dpi=300, bbox_inches="tight")
    plt.show()


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
