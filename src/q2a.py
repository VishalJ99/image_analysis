import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Define the objective functions for L1 and L2 minimisation
def l1_norm(params, x, y):
    a, b = params
    return np.sum(np.abs(a * x + b - y))


def l2_norm(params, x, y):
    a, b = params
    return np.sum((a * x + b - y) ** 2)


def main():
    # Load the data
    y_line = np.loadtxt("data/y_line.txt")
    y_outlier_line = np.loadtxt("data/y_outlier_line.txt")
    x = np.arange(1, len(y_line) + 1)

    # Initial guess for parameters a and b
    initial_params = [0, 0]

    # Perform minimisation for y_line
    result_l1_y_line = minimize(
        l1_norm, initial_params, args=(x, y_line), method="SLSQP"
    )
    result_l2_y_line = minimize(
        l2_norm, initial_params, args=(x, y_line), method="SLSQP"
    )

    # Perform minimisation for y_outlier_line
    result_l1_y_outlier_line = minimize(
        l1_norm, initial_params, args=(x, y_outlier_line), method="SLSQP"
    )
    result_l2_y_outlier_line = minimize(
        l2_norm, initial_params, args=(x, y_outlier_line), method="SLSQP"
    )

    # Print results in the terminal (COPILOT GENERATED CODE)
    print("Results for y_line (No Outliers):")
    print("L1 Norm - Slope: {:.3f}, Intercept: {:.3f}".format(*result_l1_y_line.x))
    print("L2 Norm - Slope: {:.3f}, Intercept: {:.3f}".format(*result_l2_y_line.x))

    print("\nResults for y_outlier_line (With Outliers):")
    print(
        "L1 Norm - Slope: {:.3f}, Intercept: {:.3f}".format(*result_l1_y_outlier_line.x)
    )
    print(
        "L2 Norm - Slope: {:.3f}, Intercept: {:.3f}".format(*result_l2_y_outlier_line.x)
    )

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, y_line, "o", label="Original Data (No Outliers)")
    plt.plot(x, result_l1_y_line.x[0] * x + result_l1_y_line.x[1], "-", label="L1 Fit")
    plt.plot(x, result_l2_y_line.x[0] * x + result_l2_y_line.x[1], "-", label="L2 Fit")
    plt.title("Fit for y_line Data")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, y_outlier_line, "o", label="Original Data (With Outliers)")
    plt.plot(
        x,
        result_l1_y_outlier_line.x[0] * x + result_l1_y_outlier_line.x[1],
        "-",
        label="L1 Fit",
    )
    plt.plot(
        x,
        result_l2_y_outlier_line.x[0] * x + result_l2_y_outlier_line.x[1],
        "-",
        label="L2 Fit",
    )
    plt.title("Fit for y_outlier_line Data")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/q2a.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
