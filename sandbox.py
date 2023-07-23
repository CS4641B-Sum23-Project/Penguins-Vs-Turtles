import cv2
import os
import pandas as pd
import pdb
import img_utls
#import tensorflow
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import visualizations as vis
import feature_extractions as fe

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
  
  features_dict = kwargs.get('features')
  bb_features = np.array(features_dict['t_bb_Mobilenet'].tolist())
  non_bb_features = np.array(features_dict['t_Mobilenet'].tolist())
  HOG = np.array(features_dict['t_HOG'].apply(lambda x : x.flatten()).tolist())
  
  vis.generate_visuals(training_df, validation_df, features_dict)
  print("Entering Sandbox environment")
  #pdb.set_trace()
  vis.plot_Kernel_PCA(HOG, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of HOG Translation", show_plot=True)
  vis.plot_Kernel_PCA(bb_features, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of MobileNetV2 Bounding Box Features", show_plot=True)
  vis.plot_Kernel_PCA(non_bb_features, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of MobileNetV2 Non-Bounding Box Features", show_plot=True)
  vis.plot_PCA(HOG, training_df.category_id.to_numpy(), "PCA Visualization of HOG Translation", show_plot=True)
  vis.plot_PCA(bb_features, training_df.category_id.to_numpy(), "PCA Visualization of MobileNetV2 Bounding Box Features", show_plot=True)
  vis.plot_PCA(non_bb_features, training_df.category_id.to_numpy(), "PCA Visualization of MobileNetV2 Non-Bounding Box Features", show_plot=True)
  vis.kmeans_centroids(training_df, validation_df)


  
  