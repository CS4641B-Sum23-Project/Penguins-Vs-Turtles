import cv2
import os
import pandas as pd
import pdb
import img_utls
#import tensorflow
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def edges_and_contours(training_df : pd.DataFrame, validation_df : pd.DataFrame) -> None:
  img_utls.find_contours(training_df)
  img_utls.find_edges(training_df)

  img = training_df.iloc[0].bb_image.copy()
  contours = training_df.iloc[0].contours
  edges = training_df.iloc[0].canny_edges
  cv2.imshow('Canny Edges', edges)
  cv2.waitKey(0)
  cv2.drawContours(img, contours, -1, (0,255,0), 3)
  cv2.imshow('Contours', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def enter_sandbox(training_df : pd.DataFrame, validation_df : pd.DataFrame, **kwargs) -> None:
  """ This is a sandbox function where users can experiment with the data.
      Please put any chunks of code you find valuable into it's own function
      that we can interface to. Please reference img_utls.py for how to work with
      pandas dataframes

  Args:
      training_df (pd.DataFrame): training dataframe
      validation_df (pd.DataFrame): validation dataframe
      **kwargs (dict) any aditional parameters for the sandbox
  """
  
  mobilenet_features = kwargs.get('mobilenet_features')
  print("Entering Sandbox environment")
  #pdb.set_trace()

  # get data and store in numpy array
  df = np.empty((len(training_df), 64*64*3))
  for idx, row in training_df.iterrows():
    bb_dat = row['bb_image']
    bb_dat = bb_dat.flatten()
    df[idx] = bb_dat
  print(df.shape)


  # normalize
  print(df[0])
  df = df/255.0
  print(df[0])

  kmeans = KMeans(n_clusters=2, random_state=0)
  cluster = kmeans.fit_predict(df)
  shape = kmeans.cluster_centers_.shape[1]
  

  x_data = [i for i in range(shape)]
  plt.scatter(x_data,kmeans.cluster_centers_[0], color='red', alpha=0.2, s=70)
  plt.scatter(x_data,kmeans.cluster_centers_[1], color='blue', alpha=0.2, s=50)
  plt.show()



  
  