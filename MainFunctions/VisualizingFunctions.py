import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

basecolors = list(mcolors.BASE_COLORS.values())


def visualize(X, L, V, P,dimension = 2,title = None):
    if dimension != 2:
        return
    unique_labels = np.unique(L)
    colors = {}
    for idx, label in enumerate(unique_labels):
        colors[label] = basecolors[idx]

    projected_components = np.matmul(P.T, V.T)
    color_components = [colors[label] for label in unique_labels]

    projected_samples = np.matmul(P.T, X.T)
    color_samples = [colors[label] for label in L]

    plt.scatter(x=projected_components[0, :], y=projected_components[1, :], marker="*", s=50, color=color_components)

    plt.scatter(x=projected_samples[0, :], y=projected_samples[1, :], marker=".", s=25, color=color_samples)
    if title is not None:
        plt.title(title)
    plt.show()


def visualize_many(X, L, V, P,dimension = 2,title = None):
    if dimension != 2:
        return
    unique_labels = np.unique(L)
    colors = {}
    for idx, label in enumerate(unique_labels):
        colors[label] = basecolors[idx]

    projected_components = np.matmul(P.T, V.T)
    color_components = [colors[label] for label in unique_labels]

    projected_samples = np.matmul(P.T, X.T)
    color_samples = [colors[label] for label in L]

    plt.scatter(x=projected_components[0, :], y=projected_components[1, :], marker="*", s=50, color=color_components)

    plt.scatter(x=projected_samples[0, :], y=projected_samples[1, :], marker=".", s=25, color=color_samples)

    plt.title(title)
    plt.show()