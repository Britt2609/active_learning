from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import imageio as io
import os
import datetime
from sklearn import metrics
from active.clustering import *
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


def grad_boost_classifier(X_train, X_test, y_train):

    classifier = GradientBoostingClassifier()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return y_pred


def plot_classifier_results(y_test, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.show()

    # print(classifier.coef_)
    # print(classifier.intercept_)


def active_learning_classification_test(X, y, unlabeled_fraction=0.5, random_state=42):
    """
    Active learning algorithm that iteratively selects the most informative examples to be labeled,
    and trains a classifier on the new labeled data, until all examples are labeled or a maximum number
    of iterations is reached.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target values.

    unlabeled_fraction : float, optional (default=0.5)
        The fraction of unlabeled examples to start with.

    random_state : int, optional (default=42)
        The random seed to use for splitting the data and selecting unlabeled examples.

    Returns
    -------
    clf : trained classifier
        The trained classifier.

    X_labeled : array-like of shape (n_samples_labeled, n_features)
        The labeled examples.

    y_labeled : array-like of shape (n_samples_labeled,)
        The labels of the labeled examples.

    X_unlabeled : array-like of shape (n_samples_unlabeled, n_features)
        The unlabeled examples.

    Examples
    --------
    >>> X, y = load_iris(return_X_y=True)
    >>> clf, X_labeled, y_labeled, X_unlabeled = active_learning_classification(X, y, unlabeled_fraction=0.5, random_state=42)
    """
    # Split the data into labeled and unlabeled subsets
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=unlabeled_fraction, random_state=random_state)

    # Initialize the classifier and the maximum number of iterations
    clf = SVC(kernel='linear', random_state=random_state)
    max_iterations = 2
    iteration = 0

    # Loop until all examples are labeled or the maximum number of iterations is reached
    while X_unlabeled.shape[0] > 0 and iteration < max_iterations:
        print(iteration)
        # Train the classifier on the labeled data
        clf.fit(X_labeled, y_labeled)

        # Use the active learning function to select the most informative examples from the unlabeled data
        X_new = active_learning_dbscan(X_labeled, X_unlabeled)

        # Ask the user to label the selected examples
        y_new = np.zeros(X_new.shape[0], dtype=int)  # Initialize the labels to 0
        # Here you could prompt the user to label the examples, or use another labeling method

        # Add the newly labeled examples to the labeled set
        X_labeled = np.vstack([X_labeled, X_new])
        y_labeled = np.concatenate([y_labeled, y_new])

        # Remove the newly labeled examples from the unlabeled set
        mask = np.ones(X_unlabeled.shape[0], dtype=bool)
        for i in range(X_new.shape[0]):
            mask[np.where((X_unlabeled == X_new[i]).all(axis=1))] = False
        X_unlabeled = X_unlabeled[mask]

        iteration += 1

    # Train the final classifier on all labeled data
    clf.fit(X_labeled, y_labeled)

    return clf, X_labeled, y


def active_learning(X_labeled, y_labeled, X_unlabeled, n_clusters=None, eps=0.5, min_samples=5):
    """
    Active learning algorithm that uses DBSCAN clustering to select the most informative examples from
    the unlabeled dataset, based on their distance to the labeled data.

    Parameters
    ----------
    X_labeled : pandas.DataFrame of shape (n_samples_labeled, n_features)
        The labeled data.

    y_labeled : pandas.Series of shape (n_samples_labeled,)
        The labels of the labeled data.

    X_unlabeled : pandas.DataFrame of shape (n_samples_unlabeled, n_features)
        The unlabeled data.

    n_clusters : int or None, optional (default=None)
        The maximum number of clusters that DBSCAN will try to find. If None, there is no maximum.

    eps : float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as in the same neighborhood.

    min_samples : int, optional (default=5)
        The minimum number of samples in a neighborhood for it to be considered as a core point.

    Returns
    -------
    X_new : pandas.DataFrame of shape (n_samples_new, n_features)
        The selected examples from the unlabeled data.
    """

    # Convert pandas dataframes to numpy arrays
    X_labeled = X_labeled.values
    y_labeled = y_labeled.values
    X_unlabeled = X_unlabeled.values

    # Compute pairwise distances between labeled and unlabeled data
    dist = pairwise_distances(X_unlabeled, X_labeled)

    # Compute cluster assignments using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    db.fit(dist)

    # Select the samples closest to each cluster center
    X_new = []
    for cluster_label in np.unique(db.labels_):
        if cluster_label == -1:
            continue  # ignore noise points
        cluster_mask = db.labels_ == cluster_label
        cluster_center = X_unlabeled[cluster_mask].mean(axis=0)
        dist_to_center = pairwise_distances(X_unlabeled[cluster_mask], [cluster_center])
        idx = np.argmin(dist_to_center)
        X_new.append(X_unlabeled[cluster_mask][idx])
    X_new = np.array(X_new)

    # Convert back to pandas dataframe
    X_new = pd.DataFrame(X_new, columns=X_unlabeled.columns)

    return X_new
