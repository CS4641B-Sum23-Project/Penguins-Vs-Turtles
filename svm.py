import cv2
import glob
import numpy as np
import pandas as pd
import feature_extractions as fe
import matplotlib.pyplot as plt
import tensorflow.keras.applications as applications

from sklearn import svm
from skimage import exposure
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def run_svm(training_df : pd.DataFrame, validation_df : pd.DataFrame):
    
    # get truth and image data
    train_truth = np.array(training_df['category_id'].tolist())
    valid_truth = np.array(validation_df['category_id'].tolist())

    # load and preprocess training data
    glob_dir = "images/training/*.jpg"
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    train_data = np.array(np.float32(images)/255)
    # load and preprocess validation data
    glob_dir = "images/validation/*.jpg"
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    valid_data = np.array(np.float32(images)/255)

    #if imagenet weights are being loaded, alpha must be one of `0.35`, `0.50`, `0.75`, `1.0`, `1.3` or `1.4`

    # extract features
    model = applications.MobileNetV2((224, 224, 3), 
                                     alpha = .75,
                                     pooling = 'max',
                                     include_top = False,
                                     weights = 'imagenet')
    
    train_features = model.predict(train_data)
    train_features_flat = train_features.reshape(train_features.shape[0], -1)
    
    valid_features = model.predict(valid_data)
    valid_features_flat = valid_features.reshape(valid_features.shape[0], -1)
    

    # create svm classifier
    clf = svm.SVC(kernel = 'linear')

    # train
    clf.fit(train_features_flat, train_truth)

    # make predicitons
    y_pred = clf.predict(valid_features_flat)

    # confusion matrix
    cm = confusion_matrix(valid_truth, y_pred, labels=clf.classes_)

    # plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.title("SVM - linear kernel - alpha=0.75")
    plt.show()

    accuracy = accuracy_score(y_pred, valid_truth)
    print(accuracy)
    

