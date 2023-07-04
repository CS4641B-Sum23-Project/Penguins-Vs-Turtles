import os
import img_utls

def main() -> int:
  '''
    Entry point for the program.
  '''
  
  if not os.path.exists(img_utls.IMG_DATA_PKL_PTH):
    print("Creating img_data.pkl")
    output_path, training_df, validation_df = \
       img_utls.create_img_data_pickle(keep_raw_images=True)
    print("Done.")
  else:
    print("Loading img_data.pkl")
    training_df, validation_df = \
      img_utls.load_img_data_pkl(img_utls.IMG_DATA_PKL_PTH)
    print("Done.")
  
  # img_utls.save_bounding_box_images(training_df,   BB_TRAIN_IMAGES_DIR)
  # img_utls.save_bounding_box_images(validation_df, BB_VALID_IMAGES_DIR)
  
  return 0

if __name__ == '__main__':
  ret = main()
  
  exit(ret)