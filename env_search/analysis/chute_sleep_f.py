import os
import fire
import numpy as np

from matplotlib import pyplot as plt


def plot_chute_sleep_f():
    x = np.linspace(0, 15, 100)
    # y = 2 * x**2 + 50
    y = 2 * x**2 + 50
    # y_down = y - y * 0.5
    # y_up = y + y * 0.5
    delays = np.random.exponential(100)
    print(delays)
    y_up = y + delays
    plt.plot(x, y, label="f(x) = 2x^2 + 50", color='blue')
    # y = 2 * x**2.5 + 50
    # plt.plot(x, y, label="f(x) = 2x^2.5 + 50", color='red')
    plt.fill_between(x, y, y_up
                     , color='blue', alpha=0.5)
    plt.xlabel("Centroid Distance", fontsize=25)
    plt.ylabel("Chute Sleep Time", fontsize=25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chute_sleep_f.png")
    plt.close()


    delays = np.random.exponential(100, 10000)
    plt.hist(delays, bins=10, density=True)
    plt.xlabel("Delay", fontsize=25)
    plt.ylabel("Density", fontsize=25)
    plt.title("Delay Distribution", fontsize=25)
    plt.tight_layout()
    plt.savefig("delay_distribution.png")
    plt.close()


if __name__ == '__main__':
    fire.Fire(plot_chute_sleep_f)
