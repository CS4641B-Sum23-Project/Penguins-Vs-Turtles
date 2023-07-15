import pandas as pd
from skimage import io, color, exposure
import numpy as np
import skimage.feature as ski_feat
import tensorflow as tf
import keras.applications as applications
import img_utls
import keras.preprocessing.image as image
import os
from typing import Tuple
import pickle



class Feature_Extractor:
  mobilenet_features_pkl = os.path.join(img_utls.DATA_DIR, 'mobilnet_features.pkl')
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
    
    hog_images = self.df['gray_image'].apply(_apply_hog)
    if modify_df:
      self.df['hog_images'] = hog_images
    
    return hog_images

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
    self._load_mobilenet()
    model = self.mobilenet_model
    converted_images = _244x244_images.apply(_convert_to_mobilenet_format)
    
    features = converted_images.apply(_extract_features, model=model)

    return features
    
    
  def _load_mobilenet(self) -> None:
    if not self.mobilenet_model is None: return
    self.mobilenet_model = applications.MobileNetV2(
                                          (244,244,3), 
                                          include_top=False, 
                                          weights='imagenet')
  def save_mobilenet_features(self) -> Tuple:
    """ Save the mobilenet features to pickle file

    Returns:
        Tuple: (filepath, features)
    """
    with open(self.mobilenet_features_pkl, 'wb') as f:
      features = self.extract_features_with_mobilenet()
      pickle.dump(features, f)
    
    return self.mobilenet_features_pkl, features
  def load_mobilenet_features(self) -> np.ndarray:
    """ Load the mobilenet features that were extracted

    Returns:
        np.ndarray: Mobilenet features
    """
    if not os.path.exists(self.mobilenet_features_pkl):
      path, features = self.save_mobilenet_features()
      return features
    
    with open(self.mobilenet_features_pkl, 'rb') as f:
      features = pickle.load(f)
    
    return features
    