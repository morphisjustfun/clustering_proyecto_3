import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import BallTree

from tqdm import tqdm


class DBSCAN:
    """
    eps:  the maximum distance between two points to be consider near
    distance: one of distances defined by sklearn 'pairwise.distance_metrics'
    minPts: the minimum number of neighbors for a point to be in a dense region (otherwise is noise)


    sources:
    https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.DistanceMetric.html
    """

    def __init__(self, eps, distance, minPts, p=None) -> None:
        self.eps = eps
        self.distance = distance
        self.minPts = minPts
        self.p = p
        self.label = None
        self.dataset = None
        self.tree = None

    # Remember to see dp optimization
    def precompute(self, dataset):
        pass

    def fit_predict(self, dataset) -> np.array:
        n, m = dataset.shape[0], dataset.shape[1]
        # setting labels as noise by default
        self.tree = BallTree(dataset, metric=self.distance)
        undifined, noise = -1, 0
        c = 0
        label = np.full((n, 1), undifined, dtype=int)
        for i, p in enumerate(tqdm(dataset, desc="Procesing the points in the data", leave=False)):
            if label[i] != undifined:
                continue
            NN = self.tree.query_radius([p], r=self.eps)[0]
            if len(NN) < self.minPts:
                label[i] = noise
                continue
            c += 1
            label[i] = c
            S = set(NN)
            for q in S.copy():
                if label[q] == noise: label[q] = c
                if label[q] != undifined: continue
                label[q] = c
                NN = self.tree.query_radius([dataset[q]], r=self.eps)[0]
                if len(NN) < self.minPts:
                    S |= set(NN)
        self.label = label
        self.dataset = dataset
        return label

    def plot(self):
        flatten_label = np.array(list(self.label.flatten()))
        u_label = np.unique(flatten_label)
        u_label = u_label[u_label != 0]
        for label in u_label:
            plt.scatter(self.dataset[flatten_label == label, 0], self.dataset[flatten_label == label, 1],
                        label=int(label) - 1)
        plt.legend()
