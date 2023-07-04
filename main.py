import os
import img_utls
import argparse as ap
import cv2
import pdb
def generate_data() -> str:
  """ Generate image data from scratch

  Returns:
      str: output pickle file that contains the generated data
  """
  print("Creating img_data.pkl")
  output_path, training_df, validation_df = \
      img_utls.create_img_data_pickle(keep_raw_images=True)
  
  img_utls.save_bounding_box_images(training_df,   img_utls.BB_TRAIN_IMAGES_DIR)
  img_utls.save_bounding_box_images(validation_df, img_utls.BB_VALID_IMAGES_DIR)

  return output_path

def load_data() -> None:
  print("Loading img_data.pkl")
  training_df, validation_df = \
    img_utls.load_img_data_pkl(img_utls.IMG_DATA_PKL_PTH)
  
  
  # Sandbox testing begin here #
  print("Exiting load_data()")

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

def main() -> int:
  '''
    Entry point for the program.
  '''
  parser = ap.ArgumentParser("Penguins Vs Turtles")
  parser.add_argument('-r', '--regenerate', action='store_true', help='Force regenerate data.pkl files')
  
  _args = parser.parse_args()
  regen_data = _args.regenerate
  if regen_data or not os.path.exists(img_utls.IMG_DATA_PKL_PTH):
    output_pkl_path = generate_data()
  else:
    load_data()
  
  print("Done.")

  
  return 0

if __name__ == '__main__':
  ret = main()
  
  exit(ret)