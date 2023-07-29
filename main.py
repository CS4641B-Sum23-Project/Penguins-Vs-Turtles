import os
import img_utls
import argparse as ap
import svm
from typing import Tuple

from sandbox import enter_sandbox
import feature_extractions as fe
def generate_data() -> Tuple:
  """ Generate image data from scratch

  Returns:
      Tuple: (output_path, training_df, validation_df)
  """
  print("Creating img_data.pkl")
  output_path, training_df, validation_df = \
      img_utls.create_img_data_pickle(keep_raw_images=True)
  
  img_utls.save_bounding_box_images(training_df,   img_utls.BB_TRAIN_IMAGES_DIR)
  img_utls.save_bounding_box_images(validation_df, img_utls.BB_VALID_IMAGES_DIR)

  return output_path, training_df, validation_df

def load_data() -> Tuple:
  """ Load the data from the stored pickle

  Returns:
      Tuple: (training_df, validation_df)
  """
  print("Loading img_data.pkl")
  training_df, validation_df = \
    img_utls.load_img_data_pkl(img_utls.IMG_DATA_PKL_PTH)

  return training_df, validation_df

def main() -> int:
  '''
    Entry point for the program.
  '''
  parser = ap.ArgumentParser("Penguins Vs Turtles")
  parser.add_argument('-r', '--regenerate', action='store_true', help='Force regenerate data.pkl files')
  
  _args = parser.parse_args()
  regen_data = _args.regenerate
  if regen_data or not os.path.exists(img_utls.IMG_DATA_PKL_PTH):
    output_pkl_path, training_df, validation_df = generate_data()
  else:
    training_df, validation_df = load_data()
    
  for df in [training_df, validation_df]:
    img_utls.preprocess_images(df)
  
  svm.run_svm(training_df, validation_df)
  
  '''
  FE = fe.Feature_Extractor(training_df, validation_df)
  features = FE.load_features()
  
  kwargs = {'features' : features}
  enter_sandbox(training_df, validation_df, **kwargs)
  
  print("Done.")

  '''
  return 0

if __name__ == '__main__':
  ret = main()
  
  exit(ret)