import argparse
import numpy as np
from q2_helper_funcs import fftc, ifftc, complex_soft_threshold, POCS
import matplotlib.pyplot as plt


def main(N, samples, k, lambda_, n_itrs, seed):
    # Set the random seed.
    np.random.seed(seed)

    # Generate a sparse signal in the time domain.
    t = np.zeros(N)
    nzc = np.random.rand(k) + 1
    t[np.random.choice(N, k, replace=False)] = nzc

    # Add gaussian noise.
    sigma = 0.05
    t += np.random.randn(N) * sigma

    # Visualise the original signal.
    plt.stem(t)
    plt.title("Original Sparse Signal (Time Domain)")
    plt.show()

    # Compute the frequency spectra of the signal.
    f = fftc(t)

    # Subsample the frequency spectra via uniform and random sampling.
    uniform_mask = np.zeros(N)
    uniform_mask_idxs = np.arange(N, step=N // samples)[:samples]
    uniform_mask[uniform_mask_idxs] = 1

    random_mask = np.zeros(N)
    random_idx = sorted(np.random.choice(N, samples, replace=False))
    random_mask[random_idx] = 1

    uniform_sampled_f = f * uniform_mask
    random_sampled_f = f * random_mask

    # Visualise original frequency spectra and the sampled spectras.
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].stem(np.real(f))
    ax[1].stem(np.real(uniform_sampled_f))
    ax[2].stem(np.real(random_sampled_f))

    ax[0].set_title("Original Frequency Spectrum")
    ax[1].set_title("Uniformly Sampled Frequency Spectrum")
    ax[2].set_title("Randomly Sampled Frequency Spectrum")
    plt.tight_layout()
    plt.show()

    # Reconstruct the sparse signal using the sampled frequency spectra.
    uniform_reconstructed_t, uniform_errs = POCS(uniform_sampled_f, lambda_, n_itrs, f)
    random_reconstructed_t, random_errs = POCS(random_sampled_f, lambda_, n_itrs, f)

    # Plot the error vs iteration.
    fig, ax = plt.subplots()
    ax.plot(uniform_errs, label="Uniform Sampling")
    ax.plot(random_errs, label="Random Sampling")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title("POCS Error vs Iteration")
    ax.legend()
    plt.show()

    # Compare the original, sampled and reconstructed in the time domain.
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].stem(t)
    ax[1].stem(np.real(ifftc(uniform_sampled_f)))
    ax[2].stem(np.real(uniform_reconstructed_t))

    ax[0].set_title("Original")
    ax[1].set_title("Sampled")
    ax[2].set_title("Reconstructed")
    fig.suptitle("Uniform Sampling (Time Signals)")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].stem(t)
    ax[1].stem(np.real(ifftc(random_sampled_f)))
    ax[2].stem(np.real(random_reconstructed_t))

    ax[0].set_title("Original")
    ax[1].set_title("Sampled")
    ax[2].set_title("Reconstructed")
    fig.suptitle("Random Sampling (Time Signals)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressed Sensing Reconstruction")
    parser.add_argument(
        "--N", type=int, help="Length of the signal (default: 100)", default=100
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of measurements / samples (default: 32)",
        default=32,
    )
    parser.add_argument(
        "--k", type=int, help="Sparsity level (default: 10)", default=10
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="Soft threshold regularisation parameter",
        default=0.1,
    )
    parser.add_argument(
        "--n_itrs", type=int, help="Number of iterations for POCS", default=100
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for random number generator", default=0
    )
    args = parser.parse_args()

    main(args.N, args.samples, args.k, args.lam, args.n_itrs, args.seed)
