from active.preprocess import *
from active.clustering import *
from active.classifiers import *
from active.active_learning import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

df, labels = get_labels()

# X, y = split_features_and_labels(df, labels, sample_size=2119180)
X, y, y_all_labels, X_pos, y_pos, X_neg, X_one_replaced, y_one_replaced = split_features_and_labels(df, labels, sample_size=500000)

## For clustering:
# get_cluster(X_encoded, y_all_labels)

## For clustering only positive labels
# get_cluster(X_pos, y_pos)
# X_encoded, y_encoded = encode_features(X_pos, y_pos)

X_encoded = encode_features(X)

data_full_encoded = X_encoded.join(y)

# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.9, random_state=42)
# X_train_pos, y_train_pos, X_train_neg, y_train_neg = only_positive_data(X_train, y_train)

# y_pred = grad_boost_classifier(X_train_pos, X_test, y_train_pos)

# plot_classifier_results(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.9, random_state=42)
print("splitted")

clf, X_labeled, y_labeled, X_unlabeled = active_learning_classification(X_train, y_train, unlabeled_fraction=0.5, random_state=42, eps=0.5, min_samples=5, al_method="automatic")
print("trained")
# active_learning_dbscan(X_pos, X_neg, eps=0.5, min_samples=5)
# clustering_and_AL(data_full_encoded)
y_pred = clf.predict(X_test)
print("predicted")
f1 = f1_score(y_test, y_pred)
print(f1)
