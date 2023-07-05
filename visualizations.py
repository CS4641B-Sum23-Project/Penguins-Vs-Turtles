import os
import img_utls
import cv2
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_PCA(features : np.ndarray) -> None:
  # perform PCA on the features
  pass

def dbscan_on_canny_edges(training_df : pd.DataFrame, validation_df : pd.DataFrame) -> None:

  # calculate canny_edges on 64x64 grayscale data
  data = np.array(training_df['gray_scaled'].apply(lambda x : x.flatten()).tolist())
  data = np.sum(data/255.0, axis=1, keepdims=True)

  #data = training_df['canny_edges']
  '''for i in range(data.shape[0]):
    print(data[i].shape)
  for i in range(data.shape[0]):
    cv2.imshow('Canny Edges', data[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
  dbscan = DBSCAN(eps=0.5, min_samples=2).fit(data)
  cluster = dbscan.fit_predict(data)
  labels = dbscan.labels_
  print(labels)

  # number of clusters in labels, ignorning noise if present
  ncluster = len(set(labels)) - (1 if -1 in labels else 0)
  print("Estimated # of clusters: %d" % ncluster)

def kmeans_centroids(training_df : pd.DataFrame, validation_df : pd.DataFrame) -> None:
   # get data and store in numpy array
  data = np.array(training_df['64x64'].apply(lambda x : x.flatten()).tolist())
  
  # normalize
  data = data/255.0

  kmeans = KMeans(n_clusters=2, random_state=0)
  cluster = kmeans.fit_predict(data)
  shape = kmeans.cluster_centers_.shape[1]

  x_data = np.arange(shape).tolist()
  plt.scatter(x_data,kmeans.cluster_centers_[0], color='red', alpha=0.2, s=50)
  plt.scatter(x_data,kmeans.cluster_centers_[1], color='blue', alpha=0.2, s=50)

  plt.title("KMeans Clustering of Pixel Values")
  plt.xlabel("# Pixels x Channels")
  plt.ylabel("Scaled Pixel Color Value")

  plt.show()
  print("")