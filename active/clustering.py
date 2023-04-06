import gower
import sklearn.cluster as cl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

import pandas as pd
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot


def clustering_and_AL(data):
    # Initialize the labeled and unlabeled sets
    df_copy = data.copy()
    df_copy['Label'] = df_copy['Label'].replace(1, np.nan)
    print(df_copy)
    labels = df_copy["Label"]
    labeled_set = set(data[data["Label"] == 1])
    unlabeled_set = set(data[data["Label"] == np.nan])

    # Initialize the model
    model = KMeans(n_clusters=2, random_state=0)

    # Start active learning loop
    # while len(unlabeled_set) > 0:
    for i in range(20):
        # Train the model on the labeled data
        model.fit(data[list(labeled_set)])

        # Predict labels for the unlabeled data and calculate uncertainties
        distances = euclidean_distances(data[list(unlabeled_set)], model.cluster_centers_)
        uncertainties = np.min(distances, axis=1)

        # Choose the most uncertain datapoints to label
        chosen_indices = np.argsort(-uncertainties)[:5]
        print(chosen_indices)
        print(list(unlabeled_set))
        chosen_indices = list(unlabeled_set)[chosen_indices]

        # Label the chosen datapoints and add them to the labeled set
        new_labels = np.random.randint(2, size=len(chosen_indices))
        labels[chosen_indices] = new_labels
        labeled_set.update(chosen_indices)
        unlabeled_set.difference_update(chosen_indices)
        print(new_labels)

    # Print the final labels
    print(labels)


def get_cluster(dataframe, y_all_labels):
    distance_matrix = gower.gower_matrix(dataframe)


    # Configuring the parameters of the clustering algorithm
    db_cluster = cl.KMeans(n_clusters=12)

    # Fitting the clustering algorithm
    arr = db_cluster.fit(distance_matrix)

    print("Clusters assigned are:", set(db_cluster.labels_))

    uni, counts = np.unique(arr, return_counts=True)
    d = dict(zip(uni, counts))
    print(d)

    # Adding the results to a new column in the dataframe
    dataframe["cluster"] = db_cluster.labels_
    # print(dataframe["cluster"].value_counts())

    dataframe["normal_labels"] = y_all_labels
    dataframe["label_cluster"] = dataframe["cluster"].astype(str) + " + " + dataframe["normal_labels"].astype(str)
    # print(dataframe["label_cluster"])
    print(dataframe["label_cluster"].value_counts())
    print(dataframe["normal_labels"].value_counts())


def active_learning_dbscan(X_pos, X_unlabeled, eps=0.5, min_samples=5):
    """
    Active learning algorithm that selects the most informative examples from the unlabeled dataset,
    based on their distance to the positively labeled data, using DBSCAN clustering.

    Parameters
    ----------
    X_pos : array-like of shape (n_samples_pos, n_features)
        The positively labeled data.

    X_unlabeled : array-like of shape (n_samples_unlabeled, n_features)
        The unlabeled data.

    eps : float, optional (default=0.5)
        The maximum distance between samples to be considered as part of the same neighborhood in DBSCAN.

    min_samples : int, optional (default=5)
        The minimum number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.

    Returns
    -------
    X_new : array-like of shape (n_samples_new, n_features)
        The selected examples from the unlabeled data.
    """

    # Compute pairwise distances between positive and unlabeled data
    dist = pairwise_distances(X_unlabeled, X_pos)

    # Run DBSCAN on the distances to identify clusters of points that are close to positive examples
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters = dbscan.fit_predict(dist)

    # Select the samples in the smallest cluster as the most uncertain examples
    if np.min(clusters) == -1:
        # If no clusters are found, return the unlabeled data
        X_new = X_unlabeled
    else:
        cluster_sizes = np.bincount(clusters[clusters != -1])
        smallest_cluster_idx = np.argmin(cluster_sizes)
        cluster_samples_idx = np.where(clusters == smallest_cluster_idx)[0]
        X_new = X_unlabeled[cluster_samples_idx]

    return X_new


