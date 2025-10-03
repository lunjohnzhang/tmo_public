import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_distribution(T, target_sum, t, time_sigma):
    """
    Generate a Gaussian distribution using numpy.random.normal
    such that there are T data points, the sum of the points is exactly `target_sum`,
    and it is centered at a predefined time point `t`.

    Args:
    - T (int): Total number of data points.
    - target_sum (float): Desired sum of the data points.
    - t (int): Mean of the Gaussian.
    - time_sigma (float): Standard deviation of the Gaussian.

    Returns:
    - np.ndarray: The adjusted Gaussian distribution.
    """
    # Ensure the distribution is symmetric around `t`
    gaussian = np.exp(-((np.arange(1, T + 1) - t)**2) / (2 * time_sigma**2))

    # Normalize the Gaussian to match the target sum
    normalized_gaussian = gaussian / gaussian.sum() * target_sum

    return normalized_gaussian


if __name__ == "__main__":
    # Parameters
    T = 5000
    time_sigma = 1000

    # Generate the Gaussian distribution
    gaussian_distribution = generate_gaussian_distribution(
        T=T,
        target_sum=0.2,
        t=np.random.randint(0, T),
        time_sigma=time_sigma,
    )
    gaussian_distribution_low = generate_gaussian_distribution(
        T=T,
        target_sum=0.002,
        t=np.random.randint(0, T),
        time_sigma=time_sigma,
    )

    # Plot the Gaussian distribution
    plt.plot(gaussian_distribution)
    plt.plot(gaussian_distribution_low)
    plt.plot([0.2 / T] * T, linestyle="--")
    plt.plot([0.002 / T] * T, linestyle="--")
    plt.title("Gaussian Distribution")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig("gaussian_distribution.png")
