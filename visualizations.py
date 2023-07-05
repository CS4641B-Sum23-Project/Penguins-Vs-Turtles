import os
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_PCA(features : np.ndarray) -> None:
  # perform PCA on the features
  pass

def kmeans_centroids(training_df : pd.DataFrame, validation_df : pd.DataFrame) -> None:
   # get data and store in numpy array
  df = np.empty((len(training_df), 64*64*3))
  for idx, row in training_df.iterrows():
    bb_dat = row['bb_image']
    bb_dat = bb_dat.flatten()
    df[idx] = bb_dat

  # normalize
  df = df/255.0

  kmeans = KMeans(n_clusters=2, random_state=0)
  cluster = kmeans.fit_predict(df)
  shape = kmeans.cluster_centers_.shape[1]
  

  x_data = [i for i in range(shape)]
  plt.scatter(x_data,kmeans.cluster_centers_[0], color='red', alpha=0.2, s=70)
  plt.scatter(x_data,kmeans.cluster_centers_[1], color='blue', alpha=0.2, s=50)
  plt.show()