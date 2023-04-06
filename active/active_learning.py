import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def active_learning_classification(X, y, unlabeled_fraction=0.5, random_state=42):
    """
    Active learning algorithm that iteratively selects the most informative examples to be labeled,
    and trains a classifier on the new labeled data, until all examples are labeled or a maximum number
    of iterations is reached.

    Parameters
    ----------
    X : pandas DataFrame of shape (n_samples, n_features)
        The input data.

    y : pandas Series of shape (n_samples,)
        The target values.

    unlabeled_fraction : float, optional (default=0.5)
        The fraction of unlabeled examples to start with.

    random_state : int, optional (default=42)
        The random seed to use for splitting the data and selecting unlabeled examples.

    Returns
    -------
    clf : trained classifier
        The trained classifier.

    X_labeled : pandas DataFrame of shape (n_samples_labeled, n_features)
        The labeled examples.

    y_labeled : pandas Series of shape (n_samples_labeled,)
        The labels of the labeled examples.

    X_unlabeled : pandas DataFrame of shape (n_samples_unlabeled, n_features)
        The unlabeled examples.

    Examples
    --------
    >>> X, y = load_iris(return_X_y=True)
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> clf, X_labeled, y_labeled, X_unlabeled = active_learning_classification(X, y, unlabeled_fraction=0.5, random_state=42)
    """
    # Split the data into labeled and unlabeled subsets
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=unlabeled_fraction, random_state=random_state)

    # Initialize the classifier and the maximum number of iterations
    clf = SVC(kernel='linear', random_state=random_state)
    max_iterations = 10
    iteration = 0

    # Loop until all examples are labeled or the maximum number of iterations is reached
    while X_unlabeled.shape[0] > 0 and iteration < max_iterations:
        # Train the classifier on the labeled data
        clf.fit(X_labeled, y_labeled)

        # Use the active learning function to select the most informative examples from the unlabeled data
        X_new = active_learning_dbscan(X_labeled, X_unlabeled)

        # Ask the user to label the selected examples
        y_new = pd.Series(np.zeros(X_new.shape[0], dtype=int))  # Initialize the labels to 0
        # Here you could prompt the user to label the examples, or use another labeling method

        # Add the newly labeled examples to the labeled set
        X_labeled = pd.concat([X_labeled, X_new])
        y_labeled = pd.concat([y_labeled, y_new])

        # Remove the newly labeled examples from the unlabeled set
        mask = ~(X_unlabeled.isin(X_new)).all(axis=1)
        X_unlabeled = X_unlabeled[mask]

        iteration += 1

    # Train the final classifier on all labeled data
    clf.fit(X_labeled, y_labeled)

    return clf, X_labeled, y_labeled, X_unlabeled



def active_learning_dbscan(X_pos, X_unlabeled, eps=0.5, min_samples=5):
    """
    Active learning algorithm that selects the most informative examples from the unlabeled dataset,
    based on their distance to the positively labeled data, using DBSCAN clustering.
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


