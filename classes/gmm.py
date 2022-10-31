import matplotlib.pyplot as plt
import numpy as np


def plot_gmm(x, labels):
    u_label = np.unique(labels)
    for label in u_label:
        plt.scatter(x[labels == label, 0], x[labels == label, 1], label=label)
    plt.legend()
