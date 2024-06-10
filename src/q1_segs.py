import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

def main(img_path):
    # Load the image.
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert image to numpy array.
    image_array = np.array(image)

    # Compute the Fourier Transform.
    fft_image = fft2(image_array)

    # Shift the zero-frequency component to the center.
    fft_shifted = fftshift(fft_image)

    # Compute the magnitude spectrum.
    magnitude_spectrum = np.abs(fft_shifted)

    # Log scale for better visualisation.
    log_spectrum = np.log(1 + magnitude_spectrum)

    # Display the original image and its magnitude spectrum.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image_array, cmap='gray')
    ax1.title.set_text('Original Image')
    ax1.axis('off')

    ax2.imshow(log_spectrum, cmap='gray')
    ax2.title.set_text('Magnitude Spectrum')
    ax2.axis('off')

    plt.show()

    # Define the radius of the central frequencies to keep
    rows, cols = fft_shifted.shape
    orig_i, orig_j = rows//2, cols//2  # center
    i_padding, j_padding = 20, 20  # padding

    # Create a cross mask to remove line artifacts
    mask = np.ones((rows, cols), np.uint8)
    mask[: i_padding-orig_i, orig_j-j_padding:orig_j+j_padding] = 0
    mask[i_padding-orig_i:, orig_j-j_padding:orig_j+j_padding] = 0

    mask[orig_i-i_padding:orig_i+i_padding, :orig_j-j_padding] = 0
    mask[orig_i-i_padding:orig_i+i_padding, orig_j-j_padding:] = 0

    # Fill the center with ones
    mask[orig_i-i_padding:orig_i+i_padding, orig_j-j_padding:orig_j+j_padding] = 1

    # Apply mask and inverse FFT with the corrected import
    filtered_fft_shifted = fft_shifted * mask
    filtered_image = ifft2(ifftshift(filtered_fft_shifted))
    filtered_image = np.abs(filtered_image)

    # Display the original image, the mask, and its reconstructed version
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(image_array, cmap='gray')
    ax1.title.set_text('Original Image')
    ax1.axis('off')

    ax2.imshow(mask, cmap='gray')
    ax2.title.set_text('Mask Applied')
    ax2.axis('off')

    ax3.imshow(filtered_image, cmap='gray')
    ax3.title.set_text('Reconstructed Image')
    ax3.axis('off')

    plt.show()

    # Apply an otsu thresholding to the reconstructed image
    threshold = threshold_otsu(image_array)
    binary_image = image_array > threshold
    
    # Display the original image, the reconstructed image, and the binary image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(image_array, cmap='gray')
    ax1.title.set_text('Original Image')
    ax1.axis('off')

    ax2.imshow(filtered_image, cmap='gray')
    ax2.title.set_text('Reconstructed Image')
    ax2.axis('off')
    
    ax3.imshow(binary_image, cmap='gray')
    ax3.title.set_text('Binary Image')
    ax3.axis('off')
    
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation of an image")
    parser.add_argument("image_path", type=str, help="Input file")
    args = parser.parse_args()
    
    image_path = args.image_path
    main(image_path)
    