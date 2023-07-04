import cv2
import os
import pandas as pd
import pdb
import img_utls

def enter_sandbox(training_df : pd.DataFrame, validation_df : pd.DataFrame) -> None:
  """ This is a sandbox function where users can experiment with the data.
      Please put any chunks of code you find valuable into it's own function
      that we can interface to. Please reference img_utls.py for how to work with
      pandas dataframes

  Args:
      training_df (pd.DataFrame): training dataframe
      validation_df (pd.DataFrame): Validation dataframes
  """
  print("Entering Sandbox environment")
  pdb.set_trace()
  
  img_utls.resize_images(training_df, (64,64))
  img_utls.find_contours(training_df)
  img_utls.find_edges(training_df)

  img = training_df.iloc[0].bb_image.copy()
  contours = training_df.iloc[0].contours
  edges = training_df.iloc[0].canny_edges
  cv2.drawContours(img, contours, -1, (0,255,0), 3)
  cv2.imshow('Contours', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()