from classes import KMeans, DBSCAN, plot_gmm
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def get_scores(model, X, y):
    test = [metrics.silhouette_score]
    test_sup = [normalized_mutual_info_score, adjusted_rand_score]
    test_names = ["silhouette", "normalized_mutual", "adjusted_rand_score"]
    y_pred = model.fit_predict(X)
    res = [t(X, y_pred) for t in test] + [ts(y, y_pred.ravel()) for ts in test_sup]
    df = pd.DataFrame(data=[res], columns=test_names)
    return df


def test_models(models, X, y):
    df_lst = []
    for i, model in enumerate(models):
        df = get_scores(model, X, y)
        df["model"] = models[i].__class__.__name__
        df_lst.append(df)

    df_result = pd.concat(df_lst, ignore_index=True)
    df_result.set_index(df_result['model'], inplace=True)
    df_result.drop('model', axis=1, inplace=True)
    sns.heatmap(df_result, annot=True, cmap="Blues", cbar=False)
    plt.title("Scores")
    plt.savefig(f"./plots/heatmap_scores.png")
    df_result.to_csv("./results/heatmap_scores.csv")
    plt.clf()


def graph_silhouett(x, predicted, name):
    cluster_val = np.unique(predicted)
    n_clusters = cluster_val.shape[0]
    silhouette_avg = silhouette_score(x, predicted)
    sample_silhouette_values = silhouette_samples(x, predicted)

    y_lower = 0

    for cluster in cluster_val:
        ith_cluster_silhouette_values = sample_silhouette_values[(predicted == cluster).ravel()]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(cluster) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(f"./plots/silhouette_{name}.png")
    plt.clf()


def get_data(filename="./data"):
    x = pd.read_csv(f'{filename}/dataset_tissue.txt', index_col=None)
    x = x.drop(x.columns[0], axis=1)

    labels = pd.read_csv(f'{filename}/clase.txt', skiprows=1, names=["index", "label"])
    labels = labels.drop("index", axis=1)

    y = np.array(labels["label"])
    x = np.array(x.iloc[:, :]).T
    return x, y


def test_kmean(x, y):
    kmean = KMeans(n_clusters=n_labels, norm=2)
    predicted = kmean.fit_predict(x)
    kmean.plot()
    plt.savefig("./plots/kmean.png")
    plt.clf()
    unique_y = np.unique(y)
    unique_predicted = np.array(np.unique(predicted), dtype=int)
    df = pd.DataFrame(columns=unique_y, index=unique_predicted)
    graph_silhouett(x, predicted, "kmean")
    for i in unique_predicted:
        indexes_i = np.where(predicted == i)[0]
        for j in unique_y:
            indexes_j_selected = y[indexes_i]
            df.loc[i, j] = len(indexes_j_selected[indexes_j_selected == j]) / len(indexes_i)
    # convert each row to the float type
    df = df.apply(pd.to_numeric)
    df.to_csv("./results/kmean.csv")
    print("--------------KMean-------------")
    for i in unique_predicted:
        index = pd.to_numeric(df.loc[i]).idxmax()
        print(f"Cluster {i} is {index} - {df.loc[i, index]}")
    print("-------------------------------")
    sns.heatmap(df, annot=True, cmap="Reds", cbar=False)
    plt.title("KMean results")
    plt.tight_layout()
    plt.savefig("./plots/kmean_heatmap.png")
    plt.clf()
    return kmean


def test_dbscan(x, y):
    dbscan = DBSCAN(eps=130, distance='euclidean', minPts=4)
    predicted = dbscan.fit_predict(x)
    dbscan.plot()
    plt.savefig("./plots/dbscan.png")
    plt.clf()
    unique_y = np.unique(y)
    unique_predicted = np.array(np.unique(predicted), dtype=int)
    df = pd.DataFrame(columns=unique_y, index=unique_predicted)
    df = df.apply(pd.to_numeric)
    graph_silhouett(x, predicted, "dbscan")
    for i in unique_predicted:
        indexes_i = np.where(predicted == i)[0]
        for j in unique_y:
            indexes_j_selected = y[indexes_i]
            df.loc[i, j] = len(indexes_j_selected[indexes_j_selected == j]) / len(indexes_i)

    df.to_csv("./results/dbscan.csv")
    print("----------------DBSCAN----------------")
    for i in unique_predicted:
        index = pd.to_numeric(df.loc[i]).idxmax()
        print(f"Cluster {i - 1} is {index} - {df.loc[i, index]}")
    print("-------------------------------")
    sns.heatmap(df, annot=True, cmap="Reds", cbar=False)
    plt.title("DBSCAN results")
    plt.tight_layout()
    plt.savefig("./plots/dbscan_heatmap.png")
    plt.clf()
    return dbscan


def test_gmm(x, y):
    gmm = GaussianMixture(n_components=n_labels, covariance_type='full')
    predicted = gmm.fit_predict(x)
    plot_gmm(x, predicted)
    plt.savefig("./plots/gmm.png")
    plt.clf()
    unique_y = np.unique(y)
    unique_predicted = np.array(np.unique(predicted), dtype=int)
    df = pd.DataFrame(columns=unique_y, index=unique_predicted)
    df = df.apply(pd.to_numeric)
    graph_silhouett(x, predicted, "gmm")
    for i in unique_predicted:
        indexes_i = np.where(predicted == i)[0]
        for j in unique_y:
            indexes_j_selected = y[indexes_i]
            df.loc[i, j] = len(indexes_j_selected[indexes_j_selected == j]) / len(indexes_i)
    df.to_csv("./results/gmm.csv")
    print("----------------GMM----------------")
    for i in unique_predicted:
        index = pd.to_numeric(df.loc[i]).idxmax()
        print(f"Cluster {i} is {index} - {df.loc[i, index]}")
    print("-------------------------------")
    sns.heatmap(df, annot=True, cmap="Reds", cbar=False)
    plt.title("GMM results")
    plt.tight_layout()
    plt.savefig("./plots/gmm_heatmap.png")
    plt.clf()
    return gmm


def test_agglomerative_clustering(x):
    clusters = hierarchy.linkage(x, method="ward")
    plt.figure(figsize=(8, 6))
    hierarchy.dendrogram(clusters, leaf_font_size=0)
    plt.axhline(396, color='crimson')
    plt.ylabel("Height")
    plt.savefig("./plots/agglomerative_clustering.png")
    plt.clf()


if __name__ == '__main__':
    np.random.seed(42)
    x, y = get_data()
    n_labels = len(np.unique(y))

    pca = PCA(n_components=0.90, svd_solver='full')
    x = pca.fit_transform(x)

    kmean = test_kmean(x.copy(), y)
    dbscan = test_dbscan(x.copy(), y)
    gmm = test_gmm(x.copy(), y)
    test_agglomerative_clustering(x.copy())

    models = [kmean, dbscan, gmm]
    test_models(models, x.copy(), y)
