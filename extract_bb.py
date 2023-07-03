import os
import cv2
import pandas as pd
import numpy as np
from typing import List, Iterable, Tuple

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_ANNOTE_PTH = os.path.join(SCRIPT_DIR, 'train_annotations')
VALID_ANNOTE_PTH = os.path.join(SCRIPT_DIR, 'valid_annotations')
IMAGE_DIR        = os.path.join(SCRIPT_DIR, 'images')
TRAIN_IMAGES_DIR = os.path.join(IMAGE_DIR, 'training')
VALID_IMAGES_DIR = os.path.join(IMAGE_DIR, 'validation')
EXTRACTED_BB_DIR = os.path.join(IMAGE_DIR, 'extracted')

BB_TRAIN_IMAGES_DIR = os.path.join(EXTRACTED_BB_DIR, 'training')
BB_VALID_IMAGES_DIR = os.path.join(EXTRACTED_BB_DIR, 'validation')



def ingest_annotations(training   : str, 
                       validation : str) -> Iterable[pd.DataFrame]:
  """ Ingest annotations and return the two dataframes

  Args:
      training (str): Training Annotations file path
      validation (str): Validation Annotations file path

  Returns:
      Iterable[pd.DataFrame]: (training_df, validation_df)
  """
  
  columns = ['category_id', 'bbox', 'area', 'iscrowd']
  return [pd.read_json(x, orient='records')[columns] for x in (training, validation)]

def extract_bb_images(training_df : pd.DataFrame, validation_df : pd.DataFrame,
                      save_bb_images : bool = False, add_raw_img_to_df : bool = False) -> None:
  """ Extract the images from the bounding boxes defined in the annotations

  Args:
      training_df (pd.DataFrame): Training annotations dataframe
      validation_df (pd.DataFrame): Validation annotations dataframe
      save_bb_images (bool, optional): If True, save the bounding box images to disk. 
                                       Defaults to False.
      add_raw_img_to_df (bool, optional): If True, also save the raw image to the dataframe.
                                          Defaults to False.
    
  """
  

  dirs = [TRAIN_IMAGES_DIR, VALID_IMAGES_DIR]
  extract_out = [BB_TRAIN_IMAGES_DIR, BB_VALID_IMAGES_DIR]
  dfs  = [training_df, validation_df]

  for extract_output_dir, directory, df in zip(extract_out, dirs, dfs):
    images    = []
    bb_images = []
    
    for df_index, row in df.iterrows():
      file_name = f"image_id_{df_index:03d}.jpg"
      file_path = os.path.join(directory, file_name)
      ex_out_path = os.path.join(extract_output_dir, file_name)
      
      
      image = cv2.imread(file_path)

      extracted_image = extract_bb_image(image, row.bbox)

      if save_bb_images:
        if not os.path.isdir(extract_output_dir):
          os.makedirs(extract_output_dir, exist_ok=True)
          
        cv2.imwrite(ex_out_path, extracted_image)
      
      if add_raw_img_to_df:
        images.append(image)
      bb_images.append(extracted_image)
      
    # End Inner Loop

    if add_raw_img_to_df:
      df['raw_image'] = images
    df['bb_image']  = bb_images
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

  extracted_image = image[top_left[1]:bottom_right[1],
                          top_left[0]:bottom_right[0]].copy()

  return extracted_image
def main() -> int:
  '''
    Entry point for the program.
  '''
  training_df, validation_df = ingest_annotations(
    TRAIN_ANNOTE_PTH, VALID_ANNOTE_PTH)
  
  extract_bb_images(training_df, validation_df, save_bb_images=True)
  
  
  return 0


if __name__ == '__main__':
  ret = main()
  
  exit(ret)