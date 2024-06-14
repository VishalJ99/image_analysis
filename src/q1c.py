import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import median
import argparse


def segment_coins(image):
    # Make a copy to avoid modifying the original image.
    img = image.copy()
    plt.imshow(img, cmap="gray")
    plt.title("0) Original Image")
    plt.axis("off")
    plt.show()

    # Median filter the intensity row wise to remove line artifacts.
    img_copy = img.copy()
    for idx in range(1, img.shape[0]):
        row_filtered = median(img_copy[idx, :], np.ones(10))
        img[idx, :] = row_filtered

    plt.imshow(img, cmap="gray")
    plt.title("1) Row Wise Median Filtered Image")
    plt.axis("off")
    plt.show()

    # Use Canny edge detector on the filtered image.
    edges = cv2.Canny(img, 100, 200)

    plt.imshow(edges, cmap="gray")
    plt.title("2) Canny Edge Detection")
    plt.axis("off")
    plt.show()

    inverted_edges = ~edges  # Inverting the edges

    # Slight dilation to connect the edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inverted_edges = ~cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)

    plt.imshow(inverted_edges, cmap="gray")
    plt.title("2b) Dilated Inverted Canny Edges")
    plt.axis("off")
    plt.show()

    # Label the connected components.
    labels = label(inverted_edges, connectivity=1)

    plt.imshow(labels, cmap="nipy_spectral")
    plt.title("Connected Components")
    plt.axis("off")
    plt.show()

    # Remove the largest connected component (background).
    regions = regionprops(labels)
    region_areas = {region.label: region.area for region in regions}
    largest_component = max(region_areas, key=region_areas.get)
    labels[labels == largest_component] = 0

    # Binarise the labels to get the mask.
    mask = np.zeros_like(labels)
    mask[labels > 0] = 1
    plt.imshow(mask, cmap="gray")
    plt.title("Mask after removing the largest component")
    plt.axis("off")
    plt.show()

    # Normalised Edges.
    edges = (edges / edges.max()).astype(np.int64)

    # Add edges to mask.
    mask += edges
    mask[mask > 1] = 1
    plt.imshow(mask, cmap="gray")
    plt.title("Mask + Edges")
    plt.axis("off")
    plt.show()

    # Do a morphological closing to connect the coins.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    plt.imshow(mask, cmap="gray")
    plt.title("Morphological Closing")
    plt.axis("off")
    plt.show()

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
    plt.imshow(mask, cmap="gray")
    plt.title("Median Filtered Mask")
    plt.axis("off")
    plt.show()

    # Get rid of all small connected components.
    labels = label(mask)
    regions = regionprops(labels)
    for region in regions:
        if region.area < 100:
            labels[labels == region.label] = 0
    plt.imshow(labels, cmap="nipy_spectral")
    plt.title("Mask after removing small connected components")
    plt.axis("off")
    plt.show()
    return labels


def select_coin_segs(labels):
    # Get the region properties of the labels.
    props = regionprops(labels)

    # Get a list of centroid coordinates.
    centroids_i = np.array([region.centroid[0] for region in props])
    centroids_j = np.array([region.centroid[1] for region in props])

    # Find centroid bins that have 6 elements in the x direction and 4
    # elements in the y direction.
    i_bins = [0]
    j_bins = [0]

    for i in range(0, labels.shape[0], 10):
        l_bin = i_bins[-1]
        r_bin = i

        # Count number of elements in the current bin.
        count = centroids_i[(centroids_i >= l_bin)
                            & (centroids_i < r_bin)].shape[0]

        if count == 6:
            i_bins.append(r_bin)

    for j in range(0, labels.shape[1], 10):
        l_bin = j_bins[-1]
        r_bin = j

        # Count number of elements in the current bin.
        count = centroids_j[(centroids_j >= l_bin)
                            & (centroids_j < r_bin)].shape[0]

        if count == 4:
            j_bins.append(j)

    # Visualise centroid coordinates and the identified bins.
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    fig.suptitle("Centroid Coordinates and Identified Bins")
    ax[0].scatter(list(range(len(centroids_i))), centroids_i)
    ax[0].set_title("Centroid i Coordinate scatter")
    for bin_ in i_bins:
        ax[0].axhline(bin_, color="red")

    ax[1].scatter(list(range(len(centroids_j))), centroids_j)
    ax[1].set_title("Centroid j Coordinate scatter")
    for bin_ in j_bins:
        ax[1].axhline(bin_, color="red")

    plt.tight_layout()
    plt.show()

    # Visualise the coins and their identied bin based coordinates.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(labels, cmap="nipy_spectral")
    ax.set_title("Identified Coins and their i,j coordinates")
    ax.axis("off")

    for region in props:
        i, j = np.digitize(region.centroid[0], i_bins), np.digitize(
            region.centroid[1], j_bins
        )
        ax.text(region.centroid[1], region.centroid[0], f"{i},{j}", color="k")

    plt.show()

    # Only show the coins where i = j.
    for region in props:
        i, j = np.digitize(region.centroid[0], i_bins), np.digitize(
            region.centroid[1], j_bins
        )
        if i != j:
            labels[labels == region.label] = 0

    # Visualise the result after removing coins that do not meet the criteria
    plt.imshow(labels, cmap="nipy_spectral")
    plt.title("8) Select Coins with i = j")
    plt.axis("off")
    plt.show()

    return labels


def main(image):
    coin_mask = segment_coins(image)
    final_mask = select_coin_segs(coin_mask)

    # Visualise the final mask.
    plt.imshow(image, cmap="gray")
    
    final_mask = final_mask.astype(float)
    final_mask[final_mask == 0] = np.nan  # Only show the segmented regions.
    plt.imshow(final_mask, alpha=0.5)

    plt.title("Final Coin Segmentation")
    plt.axis("off")
    plt.savefig("output_segs/q1c.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coin segmentation algorithm.")
    parser.add_argument("image_path", type=str, help="Input file")
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    main(image)
