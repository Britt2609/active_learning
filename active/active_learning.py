import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from matplotlib import pyplot as plt
from incdbscan import IncrementalDBSCAN
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


def active_learning_classification(X, y, unlabeled_fraction=0.5, random_state=42, eps=0.5, min_samples=5, al_method = "by_hand"):
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

    """
    # Split the data into labeled and unlabeled subsets
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=unlabeled_fraction, random_state=random_state)
    print(y_unlabeled)

    # print(X_labeled)
    # print(X_unlabeled)
    # Initialize the classifier and the maximum number of iterations
    clf = SVC(kernel='linear', random_state=random_state)
    max_iterations = 10
    iteration = 0

    # dbscan = IncrementalDBSCAN(eps=0.5, min_pts=5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')

    # Loop until all examples are labeled or the maximum number of iterations is reached
    while X_unlabeled.shape[0] > 0 and iteration < max_iterations:
        # Train the classifier on the labeled data
        clf.fit(X_labeled, y_labeled)

        # Use the active learning function to select the most informative examples from the unlabeled data
        X_new = active_learning_dbscan(X_labeled, X_unlabeled, dbscan)
        print(X_new.shape)

        if al_method == "by_hand":
            y_new = pd.Series(clf.predict(X_new))  # Initialize the labels to what the model predicts

            # Ask the user to label the selected examples
            for i, x in enumerate(X_new.values):
                confirmed = False
                while not confirmed:
                    # Ask the user to confirm the label
                    print(f"Sample {i + 1}: {x}")
                    print(f"Predicted label: {y_new[i]}")
                    confirmed_label = input("Is this label correct? (y/n): ")

                    if confirmed_label.lower() == "y":
                        confirmed = True
                    elif confirmed_label.lower() == "n":
                        if y_new[i] == 0:
                            y_new[i] = 1
                        else:
                            y_new[i] = 0
                        confirmed = True
                    else:
                        print("Please only give input \"y\" for yes or \"n\" for no")

                # Add the confirmed label to the labeled set, remove from unlabeled set, update model
                X_labeled, y_labeled, X_unlabeled = update_labels(X_unlabeled, X_labeled, X_new,  y_labeled, x, y_new[i])
        else:
            y_new = pd.Series(np.zeros(X_new.shape[0], dtype=int))
            for i, x in enumerate(X_new.values):
                print(y_new.index[i])
                y_new[i] = y_unlabeled[y_new.index[i]]
            # Add the newly labeled examples to the labeled set
            # y_new =
            # X_labeled = pd.concat([X_labeled, X_new])
            # y_labeled = pd.concat([y_labeled, y_new])



        # Remove the newly labeled examples from the unlabeled set
        # mask = ~(X_unlabeled.isin(X_new)).all(axis=1)
        # X_unlabeled = X_unlabeled[mask]

        iteration += 1

    # Train the final classifier on all labeled data
    clf.fit(X_labeled, y_labeled)

    return clf, X_labeled, y_labeled, X_unlabeled


def update_labels(X_unlabeled, X_labeled, X_new,  y_labeled, x, y_new_i):

    X_labeled = pd.concat([X_labeled, pd.DataFrame(x.reshape(1, -1), columns=X_labeled.columns)])
    y_labeled = pd.concat([y_labeled, pd.Series([y_new_i])])

    # Remove the newly labeled examples from the unlabeled set
    mask = ~(X_unlabeled.isin(X_new)).all(axis=1)
    X_unlabeled = X_unlabeled[mask]

    return X_labeled, y_labeled, X_unlabeled


def active_learning_dbscan(X_labeled, X_unlabeled, dbscan, eps=0.5, min_samples=5):
    """
    Active learning algorithm that selects the most informative examples from the unlabeled dataset,
    based on their distance to the positively labeled data, using DBSCAN clustering.
    """
    # Compute pairwise distances between positive and unlabeled data
    dist = pairwise_distances(X_unlabeled, X_labeled)

    # Run DBSCAN on the distances to identify clusters of points that are close to positive examples
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
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


def plot_classifier_results(y_test, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.show()

