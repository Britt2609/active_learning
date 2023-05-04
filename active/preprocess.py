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

MAIN_PATH = "/content/active_learning/Etiennes_data"


def get_labels():
    labels = pd.read_csv(MAIN_PATH + "/labelling.csv", sep=";")
    conn_mon = pd.read_csv(MAIN_PATH + "/data_mon.csv",
                           sep="\x09")
    conn_tue = pd.read_csv(MAIN_PATH + "/data_tue.csv",
                           sep="\x09")
    conn_wed = pd.read_csv(MAIN_PATH + "/data_wed.csv",
                           sep="\x09")
    conn_thu = pd.read_csv(MAIN_PATH + "/data_thu.csv",
                           sep="\x09")
    conn_fri = pd.read_csv(MAIN_PATH + "/data_fri.csv",
                           sep="\x09")
    conn_all = pd.concat([conn_mon, conn_tue, conn_wed, conn_thu, conn_fri])

    return labels, conn_all


def split_features_and_labels(df, labels, sample_size=50000):
    # sample_size = 5000
    data = df.join(labels.set_index("uid"), on=["uid"])

    features = ["id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "duration", "orig_bytes", "resp_bytes", "proto",
                "service", "conn_state"]

    data_sample = data.sample(sample_size)
    # data_sample_small = data.sample(5)

    # Make labels binary (Malicious or Benign --> 1 or 0)
    list_labels = ['Patator - FTP', 'Patator - SSH', 'DoS - Slowloris',
                   'DoS - SlowHTTPTest', 'DoS - Hulk', 'DoS - GoldenEye', 'Heartbleed',
                   'Web Attack - Brute Force', 'Web Attack - XSS',
                   'Web Attack - SQL Injection', 'Port Scan',
                   'Infiltration - Dropbox Download', 'Infiltration - Cool Disk',
                   'DDoS - Botnet', 'DDoS - LOIC']
    y_all_labels = data_sample["Label"]

    X_pos = data_sample[data_sample["Label"] != "Benign"]
    X_neg = data_sample[data_sample["Label"] == "Benign"]
    y_pos = X_pos["Label"]

    data_copy = data_sample.copy()

    # data_sample["Label"][data_sample["Label"] == 'Benign' | data_sample["Label"] == 'Web Attack - Brute Force'] = 0
    data_copy['Label'].replace({'Web Attack - Brute Force': 0, 'Benign': 0}, inplace=True)
    data_sample['Label'].replace({'Benign': 0}, inplace=True)

    data_copy = data_copy.replace(list_labels, 1)
    data_sample = data_copy.replace(list_labels, 1)

    # Select 20% of the positive data to be put to value 0.
    mask = data_copy['Label'] == 1
    n_rows = int(len(data_copy[mask]) * 0.2)
    selected_rows = data_copy[mask].sample(n=n_rows, replace=False)

    # Modify the selected rows
    data_copy.loc[selected_rows.index, 'Label'] = 0

    X_one_replaced = data_copy[features]
    y_one_replaced = data_copy["Label"]

    X = data_sample[features]
    y = data_sample["Label"]

    print("amount of differences: ")
    print(len(data_sample["Label"].compare(data_copy["Label"])))

    return X, y, y_all_labels, X_pos, y_pos, X_neg, X_one_replaced, y_one_replaced


def encode_features(X):

    # Features to be one hot encoded
    features_onehot = ["proto", "service", "conn_state"]

    onehot_data = pd.get_dummies(X, columns=features_onehot)

    # Split up the IP adresses
    onehot_data[['IP_orig_1', 'IP_orig_2', 'IP_orig_3', 'IP_orig_4']] = onehot_data["id.orig_h"].apply(lambda x: pd.Series(str(x).split(".")))
    onehot_data[['IP_resp_1', 'IP_resp_2', 'IP_resp_3', 'IP_resp_4']] = onehot_data["id.resp_h"].apply(lambda x: pd.Series(str(x).split(".")))

    encoded_data = onehot_data.drop(columns=["id.resp_h", "id.orig_h"])
    encoded_data = encoded_data.replace("-", 0)
    cols = list(encoded_data.columns)

    for feature in cols:
        encoded_data[feature] = pd.to_numeric(encoded_data[feature], downcast="float", errors='coerce')

    encoded_data = encoded_data.fillna(0)

    return encoded_data


def only_positive_data(X_train, y_train):
    # y_train_df = pd.y_train.toframe().set_axis(["Label"])
    train_data = pd.concat([X_train, y_train])
    print(y_train)
    print(train_data)
    X_train_pos = train_data[train_data["Label"] == 1]
    X_train_neg = train_data[train_data["Label"] == 0]
    y_train_pos = X_train_pos["Label"]
    y_train_neg = X_train_neg["Label"]

    return X_train_pos, y_train_pos, X_train_neg, y_train_neg


def data_for_AL():

    pass
