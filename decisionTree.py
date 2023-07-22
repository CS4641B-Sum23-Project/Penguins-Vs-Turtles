from typing import Tuple
import json
import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse as ap
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import img_utls
from main import generate_data, load_data
import feature_extractions as fe
import cv2

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
FE = fe.Feature_Extractor(training_df)
features = FE.load_features()
# Get labels from train_annotations
with open("train_annotations",'r') as file:
    data = json.load(file)
y = [obj["category_id"] for obj in data]

def convert_features(features) -> Tuple[np.ndarray, np.ndarray]:
    # Existing code to load the features dictionary
    feature_dict = features
    
    # Extract the features from the dictionary
    bb_features = feature_dict['bb_Mobilenet']
    mobilenet_features = feature_dict['Mobilenet']
    hog_features = feature_dict['HOG']
    orb_features = feature_dict['ORB']
    edges_features = feature_dict['edges']
    contours = feature_dict['contours']

    # Run for bb_features and mobilenet_features
    X_train_bb = pd.DataFrame(bb_features.tolist()).values
    X_train_mobile = pd.DataFrame(mobilenet_features.tolist()).values

    # Run for hog_features, orb_features, edges_features
    temp1 = np.array(hog_features.to_list())
    X_train_hog = temp1.reshape(temp1.shape[0], -1)

    temp2 = np.array(orb_features.to_list())
    X_train_orb = temp2.reshape(temp2.shape[0], -1)

    temp3 = np.array(edges_features.to_list())
    X_train_edges = temp3.reshape(temp3.shape[0], -1)

    # Run for contour_features
    contours_features = contours.apply(lambda x: x[0] if len(x) > 0 else [])

    # Extract specific contour features
    contours_area = contours_features.apply(lambda c: cv2.contourArea(c) if len(c) > 0 else 0)
    contours_perimeter = contours_features.apply(lambda c: cv2.arcLength(c, closed=True) if len(c) > 0 else 0)

    # Filter out contours with area or perimeter as zero
    valid_contours = contours_features[(contours_area > 0) & (contours_perimeter > 0)]
    valid_area = contours_area[(contours_area > 0) & (contours_perimeter > 0)]
    valid_perimeter = contours_perimeter[(contours_area > 0) & (contours_perimeter > 0)]

    # Calculate Hu Moments and Compactness for valid contours
    hu_moments = valid_contours.apply(lambda c: cv2.HuMoments(cv2.moments(c)).flatten())
    compactness = valid_perimeter**2 / valid_area

    # Create a DataFrame to hold the features
    features_df = pd.DataFrame({
        'contour_area': contours_area,
        'contour_perimeter': contours_perimeter,
        'hu_moment1': hu_moments.apply(lambda x: x[0]),
        'hu_moment2': hu_moments.apply(lambda x: x[1]),
        'hu_moment3': hu_moments.apply(lambda x: x[2]),
        'hu_moment4': hu_moments.apply(lambda x: x[3]),
        'hu_moment5': hu_moments.apply(lambda x: x[4]),
        'hu_moment6': hu_moments.apply(lambda x: x[5]),
        'hu_moment7': hu_moments.apply(lambda x: x[6]),
        'compactness': compactness,
    })    
    scaler = MinMaxScaler()
    X_train_contours = scaler.fit_transform(features_df)
    return X_train_bb, X_train_mobile, X_train_hog, X_train_orb, X_train_edges, X_train_contours


def find_accuracy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier
    clf = tree.DecisionTreeClassifier()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)


bb, mobile, hog, orb, edges, contours = convert_features(features)
print("bb")
find_accuracy(bb, y)
print("mobile")
find_accuracy(mobile, y)
print("hog")
find_accuracy(hog, y)
print("orb")
find_accuracy(orb, y)
print("edges")
find_accuracy(edges, y)
print("contours")
find_accuracy(contours, y)
