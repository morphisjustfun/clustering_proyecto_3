import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class KMeans:
    def __init__(self, n_clusters, norm):
        self.n_clusters = n_clusters
        self.norm = norm
        self.n_data = 0
        self.df_training_clustered = None

    @staticmethod
    def distance_norm_ln(point1, point2, n):
        return np.linalg.norm(point1 - point2, ord=n)

    def init_points(self, df):
        initial_points = df.sample(
            self.n_clusters, replace=False).iloc[:, 0:self.n_data]
        initial_points.reset_index(drop=True, inplace=True)
        for i in range(self.n_clusters):
            initial_points.loc[i, 'cluster'] = i
        return initial_points

    def kmean(self, k, df, l, initial_points, previous_points):
        df_rows = df.shape[0]
        for row in range(df_rows):
            distances = []
            points = []
            for row_point in initial_points.index:
                distances.append(
                    self.distance_norm_ln(df.iloc[row, 0:self.n_data], initial_points.iloc[row_point, 0:self.n_data],
                                          l))
                points.append(initial_points.loc[row_point, 'cluster'])
            df.loc[row, 'cluster'] = points[distances.index(min(distances))]

        # dataframe cluster property has been updated, now we need to update the initial points
        for ncluster in range(k):
            initial_points.iloc[ncluster, 0:self.n_data] = df[df['cluster']
                                                              == ncluster].iloc[:, 0:self.n_data].mean()

        if previous_points is not None and initial_points.equals(previous_points):
            return df
        else:
            return self.kmean(k, df, l, initial_points, initial_points.copy())

    def plot(self):
        u_labels = self.df_training_clustered['cluster'].unique()
        u_labels.sort()
        for i in u_labels:
            plt.scatter(self.df_training_clustered[self.df_training_clustered['cluster'] == i].iloc[:, 0],
                        self.df_training_clustered[self.df_training_clustered['cluster'] == i].iloc[:, 1], label=int(i))
        plt.legend()

    def fit_predict(self, data):
        n_cols = data.shape[1]
        if n_cols < 2:
            raise Exception('data must have at least 2 columns')
        cols = list(range(n_cols))
        data = pd.DataFrame(data, columns=cols)
        self.n_data = n_cols
        data['cluster'] = np.nan
        initial_points = self.init_points(data)
        self.df_training_clustered = self.kmean(
            self.n_clusters, data, self.norm, initial_points, None)
        return self.df_training_clustered['cluster']
