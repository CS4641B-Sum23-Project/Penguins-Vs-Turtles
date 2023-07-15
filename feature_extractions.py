import pandas as pd
from skimage import io, color, exposure
import numpy as np
import skimage.feature as ski_feat
import tensorflow as tf
import tensorflow.keras.applications as applications
import img_utls
import tensorflow.keras.preprocessing.image as image
import os
from typing import Tuple
import pickle
import cv2


class Feature_Extractor:
  features_pickle = os.path.join(img_utls.DATA_DIR, 'features.pkl')
  def __init__(self, df : pd.DataFrame):
    """ This class is manage the different feature extractions

    Args:
        df (pd.DataFrame): Training or Validation DataFrame
    """
    self.df = df
    self.mobilenet_model  = None
  
  def extract_HOG_features(self, modify_df : bool = False) -> pd.Series:
    """ Extract Histogram of Gradient features

    Args:
        modify_df (bool, optional): This will in place add the features
                                    to the DataFrame if set to True.
                                    Defaults to False.
    """
    def _apply_hog(img : np.ndarray) -> np.ndarray:
      fd, hog_image = ski_feat.hog(img, visualize=True)
      rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))
      
      return rescaled 
    
    resized_gray = self.df['gray_image'].apply(lambda x : cv2.resize(x, (64,64)))
    hog_images = resized_gray.apply(_apply_hog)
    if modify_df:
      self.df['hog_images'] = hog_images
    
    return hog_images
  
  def extract_ORB_features(self, modify_df : bool = False) -> pd.Series:
    orb = cv2.ORB_create(nfeatures=64*64, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    
    def _apply_orb(image : np.ndarray, orb):
      points, descr = orb.detectAndCompute(image, None)
      cvt = cv2.KeyPoint_convert(points)
      orb_img = np.zeros_like(image, dtype=np.uint8) 
      for point in cvt:
        x, y = point
        orb_img[int(y), int(x)] = 255
      return orb_img
    # RGB = self.df['64x64'].apply(lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    orb_images = self.df['64x64'].apply(lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)).apply(_apply_orb, orb=orb)
    return orb_images

  def extract_features_with_mobilenet(self, modify_df : bool = False) -> pd.Series:
    """ Extract Features utilizing mobilenet

    Args:
        modify_df (bool, optional): This will in place add the features
                                    to the DataFrame if set to True.
                                    Defaults to False.
    """
    def _convert_to_mobilenet_format(_244x244_image : np.ndarray) -> tf.Tensor:
      img_array = image.img_to_array(_244x244_image)
      img_array = applications.mobilenet_v2.preprocess_input(img_array)
      img_array = tf.expand_dims(img_array, 0)
      return img_array
    
    def _extract_features(tensor : tf.Tensor, model : tf.keras.Model) -> np.ndarray:
      features = model.predict(tensor)
      return features.flatten()
    
    _244x244_images = img_utls.resize_images(self.df, (244,244))
    _244x244_nonBB_images = self.df['raw_image'].apply(lambda x : cv2.resize(x, (244,244)))
    
    self._load_mobilenet()
    model = self.mobilenet_model
    converted_images = _244x244_images.apply(_convert_to_mobilenet_format)
    
    bb_features = converted_images.apply(_extract_features, model=model)
    
    converted_images = _244x244_nonBB_images.apply(_convert_to_mobilenet_format)
    
    non_bb_features = converted_images.apply(_extract_features, model=model)

    return bb_features, non_bb_features
    
    
  def _load_mobilenet(self) -> None:
    if not self.mobilenet_model is None: return
    self.mobilenet_model = applications.MobileNetV2(
                                          (244,244,3), 
                                          include_top=False, 
                                          weights='imagenet')
  def save_features(self) -> Tuple:
    """ Save the mobilenet features to pickle file

    Returns:
        Tuple: (filepath, features)
    """
    with open(self.features_pickle, 'wb') as f:
      print("Generating ORB Features")
      orb_features = self.extract_ORB_features()
      print("Generating mobilenet features")
      bb_features, non_bb_features = self.extract_features_with_mobilenet()
      print("Generating HOG Features")
      hog_features = self.extract_HOG_features()
      
      canny_features = img_utls.find_edges(self.df)['canny_edges']
      contours      = img_utls.find_contours(self.df)['contours']
      print("Saving Features to pickle")
      features = {
        'bb_Mobilenet' : bb_features,
        'Mobilenet' : non_bb_features,
        'HOG' : hog_features,
        'ORB' : orb_features,
        'edges' : canny_features,
        'contours' : contours
      }
      pickle.dump(features, f)
    
    return self.features_pickle, features
  def load_features(self) -> dict:
    """ Load the all features that were extracted

    Returns:
        dict: Dictionary of features
    """
    if not os.path.exists(self.features_pickle):
      path, feature_dict = self.save_features()
      return feature_dict
    
    with open(self.features_pickle, 'rb') as f:
      feature_dict = pickle.load(f)
    
    return feature_dict
    