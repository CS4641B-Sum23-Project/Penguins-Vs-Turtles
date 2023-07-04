import cv2
import os
import pandas as pd
import pdb
import img_utls
#import tensorflow
import feature_extractions as fe
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
  FE = fe.Feature_Extractor(training_df)
  
  hog_features = FE.extract_HOG_features()
  for img in hog_features:
    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
  
  print("Exiting Sandbox Environment")
  # pdb.set_trace()
  
  # img_utls.find_contours(training_df)
  # img_utls.find_edges(training_df)

  # img = training_df.iloc[0].bb_image.copy()
  # contours = training_df.iloc[0].contours
  # edges = training_df.iloc[0].canny_edges
  # cv2.drawContours(img, contours, -1, (0,255,0), 3)
  # cv2.imshow('Contours', img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()