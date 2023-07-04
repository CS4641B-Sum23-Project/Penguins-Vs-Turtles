import pandas as pd
from skimage import io, color, exposure
import numpy as np
import skimage.feature as ski_feat
import tensorflow
class Feature_Extractor:
  
  def __init__(self, df : pd.DataFrame):
    """ This class is manage the different feature extractions

    Args:
        df (pd.DataFrame): Training or Validation DataFrame
    """
    self.df = df
  
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