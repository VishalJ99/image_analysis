import numpy as np
import argparse


def f(x1, x2):
    return (x1**2 / 2) + x2**2


def grad_f(x1, x2):
    return (x1, 2 * x2)


def main(lr):
    # Initial point.
    x0 = np.array((1, 1))

    # Global minimum.
    x_star = np.array((0, 0))

    # Define the error threshold.
    err_thresh = 0.01

    # Initialise the iteration count.
    iter_count = 0

    # Initialise the current point.
    xi = x0.copy()

    # Initialise the starting error.
    err = np.abs(f(*xi) - f(*x_star))
    while err > err_thresh:
        print(f"iter: {iter_count}, xi: {xi}, f(xi): {f(*xi)}, err: {err}")
        # Compute the gradient at the current point.
        grad = grad_f(*xi)

        # Update the current point.
        x1 = xi[0] - lr * grad[0]
        x2 = xi[1] - lr * grad[1]
        xi = (x1, x2)

        # Compute the error.
        err = np.abs(f(*xi) - f(*x_star))

        iter_count += 1

    print(f"iter: {iter_count}, xi: {xi}, f(xi): {f(*xi)}, err: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Descent Optimization")
    parser.add_argument(
        "--lr", type=float, default=0.5, help="Learning rate (step size)"
    )
    args = parser.parse_args()
    main(lr=args.lr)
