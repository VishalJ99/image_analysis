import numpy as np
from typing import Tuple


def complex_soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """
    Apply complex soft thresholding to the input array.

    Parameters:
    -----------
    x : np.ndarray
        The input array.

    lam : float
        The threshold value.

    Returns:
    --------
    np.ndarray
        The soft thresholded array.
    """
    res = abs(x) - lam
    return (res > 0.0) * res * x / abs(x)


def fftc(x: np.ndarray) -> np.ndarray:
    """
    Compute the centered Fast Fourier Transform (FFT) of an input array.

    Parameters:
    -----------
    x : np.ndarray
        Input array.

    Returns:
    --------
    np.ndarray
        The centered FFT of the input array.

    Notes:
    ------
    The normalisation factor is applied to ensure that the result has
    the same energy as the input array.
    """
    norm = 1 / np.sqrt(np.prod(x.shape))
    return norm * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def ifftc(y: np.ndarray) -> np.ndarray:
    """
    Compute the centered inverse fast Fourier transform (IFFT) of the
    input array.

    Parameters:
    -----------
    y : np.ndarray
        The input array.

    Returns:
    --------
    ifft : np.ndarray
        The IFFT of the input array.

    Notes:
    ------
    The normalisation factor is applied to ensure that the result has
    the same energy as the input array.
    """
    norm = np.sqrt(np.prod(y.shape))
    return norm * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(y)))


def POCS(
    y_obs: np.ndarray, lam: float, nitr: int, y_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the Projection Onto Convex Sets (POCS) algorithm for signal
    reconstruction. The POCS algorithm iteratively performs the following:
    1. Transform the observed signal to the sparse (time) domain.
    2. Apply soft thresholding to the sparse signal.
    3. Transform the sparse signal back to the frequency domain.
    4. Enforce the observed values in the frequency domain.
    5. Compute the error between the reconstructed signal and the ref signal.

    The algorithm returns the final reconstructed sparse signal and the
    error at each iteration.
    Parameters:
    -----------
    y_obs : np.ndarray
        The observed signal in the frequency domain.
    lam : float
        The regularization parameter for soft thresholding.
    nitr : int
        The number of iterations to perform.
    ref : array_like
        The reference signal for computing the error.

    Returns:
    --------
    final_s : array_like
        The reconstructed sparse signal in the time domain.
    err : array_like
        The error at each iteration.
    """

    err = np.zeros((nitr,))
    # Copy the observed signal.
    y_i = y_obs.copy()
    for i in range(nitr):
        # Go to time (sparse) domain.
        s_i = ifftc(y_i)

        if i == 0:
            # Correct for the reduced energy in the time domain.
            correction = len(y_obs) / sum(np.abs(y_obs) > 0)
            s_i = correction * s_i

        # Apply complex soft thresholding.
        s_i = complex_soft_threshold(s_i, lam)

        # Go back to freq domain.
        y_i = fftc(s_i)

        # Enforce the observed values.
        y_i = y_i * (y_obs == 0) + y_obs

        # Compute the error.
        err[i] = np.linalg.norm(y_i - y_ref)

    # Return the sparse signal and the error.
    final_s = ifftc(y_i)
    return final_s, err
