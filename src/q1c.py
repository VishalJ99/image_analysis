import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import median
import argparse

# import binary fill holes from skimage
from scipy.ndimage import binary_fill_holes


def segment_coins(image):
    # Make a copy to avoid modifying the original image.
    img = image.copy()

    # Median filter the intensity row wise to remove line artifacts.
    img_copy = img.copy()
    for idx in range(1, img.shape[0]):
        row_filtered = median(img_copy[idx, :], np.ones(10))
        img[idx, :] = row_filtered

    # Use Canny edge detector on the filtered image.
    edges = cv2.Canny(img, 100, 200)

    # Invert the edges.
    inverted_edges = ~edges

    # Slight dilation to connect the edge outlines.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inverted_edges = ~cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)

    # Label the connected components.
    labels = label(inverted_edges, connectivity=1)

    # Remove the largest connected component (background).
    regions = regionprops(labels)
    region_areas = {region.label: region.area for region in regions}
    largest_component = max(region_areas, key=region_areas.get)
    labels[labels == largest_component] = 0

    # Binarise the labels to get the mask.
    mask = np.zeros_like(labels)
    mask[labels > 0] = 1

    # Normalised Edges.
    edges = (edges / edges.max()).astype(np.int64)

    # Add edges to mask.
    mask += edges

    # Binarise the mask.
    mask[mask > 1] = 1

    # Apply binary hole filling to the mask.
    mask = binary_fill_holes(mask)

    # Apply row wise median filter to the mask.
    mask_copy = mask.copy()
    for idx in range(1, image.shape[0]):
        row_filtered = median(mask_copy[idx, :], np.ones(10))
        mask[idx, :] = row_filtered

    # Apply column wise median filter to the mask.
    mask_copy = mask.copy()
    for idx in range(1, image.shape[1]):
        col_filtered = median(mask_copy[:, idx], np.ones(10))
        mask[:, idx] = col_filtered

    # Get rid of all small connected components.
    labels = label(mask)
    regions = regionprops(labels)
    for region in regions:
        if region.area < 100:
            labels[labels == region.label] = 0

    return labels


def select_coin_segs(labels):
    # Get the region properties of the labels.
    props = regionprops(labels)

    # Get a list of centroid coordinates.
    centroids_i = np.array([region.centroid[0] for region in props])
    centroids_j = np.array([region.centroid[1] for region in props])

    # Sort the centroids based on their i and j coordinates.
    sorted_centroids_i_idx = np.argsort(centroids_i)
    sorted_centroids_j_idx = np.argsort(centroids_j)

    # Assign the a row and column number to each coin.
    for idx in range(len(sorted_centroids_i_idx)):
        # Fetch region corresponding to centroid with i coordinate.
        region_i = props[sorted_centroids_i_idx[idx]]

        # Assign it an i attribute based on its index.
        region_i.i = idx // 6

        # Update the region in the props list.
        props[sorted_centroids_i_idx[idx]] = region_i

        # Repeat the same for the j coordinate.
        region_j = props[sorted_centroids_j_idx[idx]]
        region_j.j = idx // 4
        props[sorted_centroids_j_idx[idx]] = region_j

    # Only show the coins where i = j.
    for region in props:
        i, j = region.i, region.j
        if i != j:
            labels[labels == region.label] = 0

    return labels


def main(image):
    coin_mask = segment_coins(image)
    final_mask = select_coin_segs(coin_mask)

    # Visualise the final mask.
    plt.imshow(image, cmap="gray")

    final_mask = final_mask.astype(float)
    final_mask[final_mask == 0] = np.nan  # Only show the segmented regions.
    plt.imshow(final_mask, alpha=0.5)
    plt.title("Coin Segmentation")
    plt.axis("off")
    plt.savefig("outputs/q1c_segmentation.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()

    print("[INFO] Plot saved at 'outputs/q1c_segmentation.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coin segmentation algorithm.")
    parser.add_argument("image_path", type=str, help="Input file")
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    main(image)
