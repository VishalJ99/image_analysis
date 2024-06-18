"""
From scratch implementations of the segmentation algorithm for q1a
Lung CT segmentation.
"""
import numpy as np
from collections import deque


def otsu_threshold_from_scratch(image: np.ndarray) -> int:
    """
    Following algorithm detailed in:
    https://en.wikipedia.org/wiki/Otsu%27s_method.
    Calculates the optimal threshold for a binary image using Otsu's method.
    Optimal threshold is the threshold that minimises the intra-class variance
    or equivalently maximises the inter-class variance.

    Parameters:
    ------------
    image : np.ndarray
        A 2D grayscale image.

    Returns:
    --------
    int
    """
    # Compute the histogram of the image.
    pixel_counts = np.histogram(image, bins=np.arange(257))[0]

    # Compute total number of pixels.
    total = image.size

    # Compute sum of all intensities.
    sum_total = np.dot(np.arange(256), pixel_counts)

    # Initialise variables.
    sumB, wB, wF, max_inter_class_var, threshold = 0.0, 0, 0, 0.0, 0

    # Iterate through all possible thresholds.
    for t in range(256):
        # Update background weight.
        wB += pixel_counts[t]

        # If no elements in background, skip threshold.
        if wB == 0:
            continue

        # Update foreground weight.
        wF = total - wB

        # If no elements in foreground, terminate algorithm.
        if wF == 0:
            break

        # Update background and foreground means.
        sumB += t * pixel_counts[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        # Calculate inter class variance.
        inter_class_var = wB * wF * (mB - mF) ** 2

        # Check if new maximum found.
        if inter_class_var > max_inter_class_var:
            max_inter_class_var = inter_class_var
            threshold = t

    return threshold


def label_from_scratch(image: np.ndimage, connectivity: int = 2) -> np.ndarray:
    """
    Following algorithm detailed in:
    https://en.wikipedia.org/wiki/Connected-component_labeling. Label the
    connected components in a binary image using breadth-first search (BFS).
    The function supports 4-connectivity and 8-connectivity.

    Parameters:
    ------------
    image : np.ndarray
        A 2D binary image.

    connectivity : int
        The connectivity of the connected components. Must be 1 or 2.

    Returns:
    --------
    np.ndarray
        A 2D array with the connected components labelled.
    """
    assert image.ndim == 2, "Image must be 2D."
    assert connectivity in [1, 2], "Connectivity must be 1 or 2."

    # Define relative directions for 4-connectivity.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # If connectivity is 2 add diagonal directions.
    if connectivity == 2:
        directions.extend([(-1, -1), (-1, 1), (1, 1), (1, -1)])

    # Initialise labeling and processed state arrays.
    label = 0
    labelled_array = np.zeros(image.shape, dtype=int)
    processed_array = np.zeros(image.shape, dtype=bool)

    # Helper function to check if a pixel inside image bounds and unprocessed.
    def is_valid(i, j):
        return (
            0 <= i < image.shape[0]
            and 0 <= j < image.shape[1]
            and not processed_array[i][j]
        )

    # Loop through each pixel in the image.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Check if pixel is a unprocessed foreground pixel.
            if image[i][j] == 1 and not processed_array[i][j]:
                # Mark pixel as processed.
                processed_array[i][j] = True

                # Increment label.
                label += 1

                # Label the current pixel.
                labelled_array[i][j] = label

                # Initialise queue with current pixel as root.
                queue = deque([(i, j)])  # Use deque for O(1) pop from left.

                # Perform BFS to label the connected tree of pixels.
                while queue:
                    # NOTE: Redefinition of i, j does not affect outer scope.
                    i, j = queue.popleft()

                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if is_valid(ni, nj) and image[ni][nj] == 1:
                            processed_array[ni][nj] = True
                            labelled_array[ni][nj] = label
                            queue.append((ni, nj))

    return labelled_array


def binary_fill_holes_from_scratch(input_image: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary image.
    Implementation of algorithm described in:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html.

    Parameters:
    ------------
    input_image : np.ndarray
        A 2D binary image.

    Returns:
    --------
    np.ndarray
        A binary image with holes filled.
    """
    # Ensure the input is a binary image.
    binary_image = np.where(input_image > 0, 1, 0)

    # Invert the image to make holes foreground.
    inverted_image = 1 - binary_image

    # Label connected components.
    labelled_image = label_from_scratch(inverted_image)
    num_features = np.amax(labelled_image)

    # Create an array that will be 1 at the borders and 0 elsewhere.
    border_mask = np.zeros_like(binary_image)
    border_mask[:, 0] = border_mask[:, -1] = 1
    border_mask[0, :] = border_mask[-1, :] = 1

    # Label connected components on the border.
    border_labels = np.unique(labelled_image * border_mask)

    # Determine which components in labelled_image are not touching the border.
    hole_labels = np.setdiff1d(np.arange(1, num_features + 1), border_labels)

    # Fill holes: components that are labelled and not touching the border.
    for label_num in hole_labels:
        binary_image[labelled_image == label_num] = 1

    return binary_image
