import os
import cv2
import pandas as pd
import numpy as np
import pickle
from typing import List, Iterable, Tuple

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_ANNOTE_PTH = os.path.join(SCRIPT_DIR, 'train_annotations')
VALID_ANNOTE_PTH = os.path.join(SCRIPT_DIR, 'valid_annotations')
IMAGE_DIR        = os.path.join(SCRIPT_DIR, 'images')
TRAIN_IMAGES_DIR = os.path.join(IMAGE_DIR, 'training')
VALID_IMAGES_DIR = os.path.join(IMAGE_DIR, 'validation')
EXTRACTED_BB_DIR = os.path.join(IMAGE_DIR, 'extracted')
DATA_DIR         = os.path.join(SCRIPT_DIR, 'data')


BB_TRAIN_IMAGES_DIR = os.path.join(EXTRACTED_BB_DIR, 'training')
BB_VALID_IMAGES_DIR = os.path.join(EXTRACTED_BB_DIR, 'validation')
IMG_DATA_PKL_NAME   = 'image_data.pkl'
IMG_DATA_PKL_PTH    = os.path.join(DATA_DIR, IMG_DATA_PKL_NAME)



def ingest_annotations(training   : str = TRAIN_ANNOTE_PTH, 
                       validation : str = VALID_ANNOTE_PTH) -> Iterable[pd.DataFrame]:
  """ Ingest annotations and return the two dataframes

  Args:
      training (str): Training Annotations file path
      validation (str): Validation Annotations file path

  Returns:
      Iterable[pd.DataFrame]: (training_df, validation_df)
  """
  
  columns = ['category_id', 'bbox', 'area', 'iscrowd']
  return [pd.read_json(x, orient='records')[columns] for x in (training, validation)]

def _save_extracted_image(row : pd.Series, dir : str) -> None:
  """ This function is meant to be utilized by a pd.DataFrame.apply()
      This will save out the extracted images to the relevant output directory

  Args:
      row (pd.Series): Row of the dataframe
      dir (str): Extracted output directory
  """
  cv2.imwrite(
    os.path.join(dir, f"image_id_{row.name:03d}.jpg"),
    row.bb_image
  )

def extract_bb_images(training_df : pd.DataFrame = None, validation_df : pd.DataFrame = None,
                      add_raw_img_to_df : bool = False) -> Tuple[pd.DataFrame]:
  """ Extract the images from the bounding boxes defined in the annotations

  Args:
      training_df (pd.DataFrame): Training annotations dataframe
      validation_df (pd.DataFrame): Validation annotations dataframe
      save_bb_images (bool, optional): If True, save the bounding box images to disk. 
                                       Defaults to False.
      add_raw_img_to_df (bool, optional): If True, also save the raw image to the dataframe.
                                          Defaults to False.
  Returns:
      tuple: (training_df, validation_df)  
  """
  
  
  ### Helper Functions ###
  def _apply_extract(row : pd.Series) -> np.ndarray:
    return extract_bb_image(row.raw_image, row.bbox)
  
  def _load_image(row : pd.Series, dir : str) -> np.ndarray:
    return cv2.imread(
      os.path.join(dir, f"image_id_{row.name:03d}.jpg")
    )
  
  ### Begin
  
  # Ingest annotations if one isn't provided
  if training_df is None or validation_df is None:
    _train, _valid = ingest_annotations()
    if not training_df:
      training_df = _train
    if not validation_df:
      validation_df = _valid

  # Create iterables
  dirs        = [TRAIN_IMAGES_DIR, VALID_IMAGES_DIR]
  dfs         = [training_df, validation_df]

  for directory, df in zip(dirs, dfs):
    df['raw_image'] = df.apply(_load_image, axis=1, dir=directory)
    df['bb_image']  = df.apply(_apply_extract, axis=1)
    
    if not add_raw_img_to_df:
      df.drop('raw_image', axis=1)
  
  return (training_df, validation_df)
  # End Outer Loop
  
  
def convert_bb_coords(bbox : List[int]) -> Tuple[Tuple]:
  """ Convert the bounding box array from the annotation
      into the format that is expected for indexing

  Args:
      bbox (List[int]): [x_top_left, y_top_left, x_span, y_span]

  Returns:
      tuple: (Top_left_Coord, Bottom_Right_Coord)
  """
  
  top_left      = tuple(bbox[:2])
  bottom_right  = (bbox[0] + bbox[2], bbox[1] + bbox[3])
  
  return (top_left, bottom_right)
  
def extract_bb_image(image : np.ndarray, bb : List[int]) -> np.ndarray:
  """ Extract the bounding box from the given image

  Args:
      image (np.ndarray): Original image
      bb (List[int]): Bounding box data from the annotations

  Returns:
      np.ndarray: Bounding box image
  """

  top_left, bottom_right = convert_bb_coords(bb)

  # Copy may not be needed if image processing calls returns a
  # new array rather than modifying the underlying array
  
  extracted_image = image[top_left[1]:bottom_right[1],
                          top_left[0]:bottom_right[0]].copy()

  return extracted_image

def create_img_data_pickle(keep_raw_images : bool = False) -> Tuple:
  """ Create an image data pickle file.

  Args:
      keep_raw_images (bool, optional): Flag to keep the raw image data in the dataframes. 
                                        Defaults to False.

  Returns:
      str: filepath to the pickle file saved
  """
  if keep_raw_images:
    training_df, validation_df = extract_bb_images(add_raw_img_to_df=keep_raw_images)
  else:
    training_df, validation_df = extract_bb_images()
  
  if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
  
  output_path = IMG_DATA_PKL_PTH
  
  with open(IMG_DATA_PKL_PTH, 'wb') as f:
    _dict = {'training_df' : training_df, 'validation_df' : validation_df}
    pickle.dump(_dict, f)
  
  return (output_path, training_df, validation_df)

def save_bounding_box_images(df : pd.DataFrame, output_dir : str) -> None:
  """ Save the bounding box images to disk

  Args:
      df (pd.DataFrame): Dataframe for either training or validation data.
      output_dir (str): Training or validation extraction directory.
  """
    
  if not os.path.isdir(output_dir):
      os.makedirs(output_dir, exist_ok=True)
  df.apply(_save_extracted_image, axis=1, dir=output_dir)

def load_img_data_pkl(path : str) -> Tuple[pd.DataFrame]:
  """ Load the given pickle and return the training and validation
      dataframe in a tuple

  Args:
      path (str): filepath to img_data.pkl

  Returns:
      Tuple[pd.DataFrame]: (training_df, validation_df)
  """
  
  with open(path, 'rb') as f:
    _dict = pickle.load(f)

  return tuple(_dict.values())
  
def find_contours(df : pd.DataFrame) -> pd.DataFrame:
  """ Find the contours in the given image.

  Args:
      df (pd.DataFrame): Dataframe with image data.

  Returns:
      pd.DataFrame: Dataframe with an added column of contours
  """
  def apply_contours(image : np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
    countours, hierarchy = cv2.findContours(
      thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    return countours
  
  series_contours = df['bb_image'].apply(apply_contours, axis=1) 
    

def resize_images(df : pd.DataFrame, size : tuple) -> pd.DataFrame:
  """ Resize the images to the defined size

  Args:
      df (pd.DataFrame): _description_
      size (tuple): Width x Height for the new shape

  Returns:
      pd.DataFrame: _description_
  """
  def apply_resize( img : np.ndarray) -> np.ndarray:
    return cv2.resize(img, size)
  
  df['bb_image'] = df['bb_image'].apply(apply_resize)