from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio as io
import os


file_path = "Etiennes_data/Monday-WorkingHours/ssh.log"


def read_data(path):
    """Read Zeek csv."""
    df = pd.read_csv(path)
    print(df[:10])
    df["ts"] = pd.to_datetime(df["ts"])
    if "ts_" in df.columns:
        df["ts_"] = pd.to_datetime(df["ts_"])
    if "duration" in df.columns:
        df["duration"] = pd.to_timedelta(df["duration"])
    print(df)
    return df


dataset = read_data(file_path)
