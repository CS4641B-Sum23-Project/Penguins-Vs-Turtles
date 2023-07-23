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
import pdb
from random_forest import RandomForest
import multiprocessing as mp
import pickle

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
FE = fe.Feature_Extractor(training_df, validation_df)
features = FE.load_features()

y_train = training_df['category_id'].to_numpy()
y_test = validation_df['category_id'].to_numpy()

def convert_features(features) -> Tuple[np.ndarray, np.ndarray]:
    # Existing code to load the features dictionary
    feature_dict = features
    
    # Extract the features from the dictionary
    t_bb_features = feature_dict['t_bb_Mobilenet']
    t_mobilenet_features = feature_dict['t_Mobilenet']
    t_hog_features = feature_dict['t_HOG']
    t_orb_features = feature_dict['t_ORB']
    t_edges_features = feature_dict['t_edges']
    t_contours = feature_dict['t_contours']
    
    v_bb_features = feature_dict['v_bb_Mobilenet']
    v_mobilenet_features = feature_dict['v_Mobilenet']
    v_hog_features = feature_dict['v_HOG']
    v_orb_features = feature_dict['v_ORB']
    v_edges_features = feature_dict['v_edges']
    v_contours = feature_dict['v_contours']
    

    # Run for bb_features and mobilenet_features
    X_train_bb      = np.array(t_bb_features.tolist())
    X_train_mobile  = np.array(t_mobilenet_features.tolist())
    y_test_bb       = np.array(v_bb_features.tolist())
    y_test_mobile   = np.array(v_mobilenet_features.tolist())

    # Run for hog_features, orb_features, edges_features
    temp1       = np.array(t_hog_features.to_list())
    X_train_hog = temp1.reshape(temp1.shape[0], -1)
    temp2       = np.array(v_hog_features.to_list())
    y_test_hog  = temp2.reshape(temp2.shape[0], -1)

    temp3       = np.array(t_orb_features.to_list())
    X_train_orb = temp3.reshape(temp3.shape[0], -1)
    temp4       = np.array(v_orb_features.to_list())
    y_test_orb  = temp4.reshape(temp4.shape[0], -1)

    temp5 = np.array(t_edges_features.to_list())
    X_train_edges = temp5.reshape(temp5.shape[0], -1)
    temp6 = np.array(v_edges_features.to_list())
    y_test_edges = temp6.reshape(temp6.shape[0], -1)

    # Run for contour_features
    t_contours_features = t_contours.apply(lambda x: x[0] if len(x) > 0 else [])
    v_contours_features = v_contours.apply(lambda x: x[0] if len(x) > 0 else [])

    # Extract specific contour features
    t_contours_area = t_contours_features.apply(lambda c: cv2.contourArea(c) if len(c) > 0 else 0)
    t_contours_perimeter = t_contours_features.apply(lambda c: cv2.arcLength(c, closed=True) if len(c) > 0 else 0)
    
    v_contours_area = v_contours_features.apply(lambda c: cv2.contourArea(c) if len(c) > 0 else 0)
    v_contours_perimeter = v_contours_features.apply(lambda c: cv2.arcLength(c, closed=True) if len(c) > 0 else 0)

    # Filter out contours with area or perimeter as zero
    t_valid_contours = t_contours_features[(t_contours_area > 0) & (t_contours_perimeter > 0)]
    t_valid_area = t_contours_area[(t_contours_area > 0) & (t_contours_perimeter > 0)]
    t_valid_perimeter = t_contours_perimeter[(t_contours_area > 0) & (t_contours_perimeter > 0)]
    
    v_valid_contours = v_contours_features[(v_contours_area > 0) & (v_contours_perimeter > 0)]
    v_valid_area = v_contours_area[(v_contours_area > 0) & (v_contours_perimeter > 0)]
    v_valid_perimeter = v_contours_perimeter[(v_contours_area > 0) & (v_contours_perimeter > 0)]

    # Calculate Hu Moments and Compactness for valid contours
    t_hu_moments = t_valid_contours.apply(lambda c: cv2.HuMoments(cv2.moments(c)).flatten())
    t_compactness = t_valid_perimeter**2 / t_valid_area

    v_hu_moments = v_valid_contours.apply(lambda c: cv2.HuMoments(cv2.moments(c)).flatten())
    v_compactness = v_valid_perimeter**2 / v_valid_area
    # Create a DataFrame to hold the features
    t_features_df = pd.DataFrame({
        'contour_area': t_contours_area,
        'contour_perimeter': t_contours_perimeter,
        'hu_moment1': t_hu_moments.apply(lambda x: x[0]),
        'hu_moment2': t_hu_moments.apply(lambda x: x[1]),
        'hu_moment3': t_hu_moments.apply(lambda x: x[2]),
        'hu_moment4': t_hu_moments.apply(lambda x: x[3]),
        'hu_moment5': t_hu_moments.apply(lambda x: x[4]),
        'hu_moment6': t_hu_moments.apply(lambda x: x[5]),
        'hu_moment7': t_hu_moments.apply(lambda x: x[6]),
        'compactness': t_compactness,
    })
    
    v_features_df = pd.DataFrame({
        'contour_area': v_contours_area,
        'contour_perimeter': v_contours_perimeter,
        'hu_moment1': v_hu_moments.apply(lambda x: x[0]),
        'hu_moment2': v_hu_moments.apply(lambda x: x[1]),
        'hu_moment3': v_hu_moments.apply(lambda x: x[2]),
        'hu_moment4': v_hu_moments.apply(lambda x: x[3]),
        'hu_moment5': v_hu_moments.apply(lambda x: x[4]),
        'hu_moment6': v_hu_moments.apply(lambda x: x[5]),
        'hu_moment7': v_hu_moments.apply(lambda x: x[6]),
        'compactness': v_compactness,
    })    
    scaler = MinMaxScaler()
    X_train_contours    = scaler.fit_transform(t_features_df)
    y_test_contours     = scaler.fit_transform(v_features_df)
    
    t_gray = np.array(training_df['64x64'].apply(lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).flatten()).tolist())
    v_gray = np.array(validation_df['64x64'].apply(lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).flatten()).tolist())

    return X_train_bb, X_train_mobile, X_train_hog, X_train_orb, X_train_edges, X_train_contours, t_gray, \
        y_test_bb, y_test_mobile, y_test_hog, y_test_orb, y_test_edges, y_test_contours, v_gray


def find_accuracy(X_train, y_train, X_test, y_test):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier
    clf = tree.DecisionTreeClassifier()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)



def mobile_net_bb_hp(x_train, y_train, x_test, y_test) -> None:
    _args = []
    A = np.arange(3, 16)
    B = np.arange(3, 16)
    C = np.arange(.3, 1.025, .025)
    
    X,Y,Z= np.meshgrid(A, B, C)
    iterator = np.nditer([X,Y,Z])
    _args = []
    
    for x in iterator:
        _args.append((int(x[0]), int(x[1]), float(x[2])))
    results = []

    num_proc = mp.cpu_count() - 2
    fit_RF(_args[0])
    pool = mp.Pool(processes=num_proc)
    results = pool.map(fit_RF, _args)
    results = sorted(results, key=lambda x:x[0])
    results = results[::-1]
    
    with open('mobile_net_bb_rf.pkl', 'wb') as f:
        pickle.dump(results, f)

    
def fit_RF(my_tuple : tuple) -> float:

    (estimators, max_depth, max_features) = my_tuple
    RF = RandomForest(int(estimators), int(max_depth), float(max_features), random_seed=(4641 + 7641))
    RF.fit(X_train, y_train)
    acc = RF.OOB_score(X_test, y_test)

    return [acc, estimators, max_depth, max_features]


t_bb, t_mobile, t_hog, t_orb, t_edges, t_contours, t_gray, \
v_bb, v_mobile, v_hog, v_orb, v_edges, v_contours, v_gray, \
    = convert_features(features)

#mobile_net_bb_hp(t_bb, y_train, v_bb, y_test)

print("grayscale")
find_accuracy(t_gray, y_train, v_gray, y_test)
print("bb")
find_accuracy(t_bb, y_train, v_bb, y_test)
print("mobile")
find_accuracy(t_mobile, y_train, v_mobile, y_test)
print("hog")
find_accuracy(t_hog, y_train, v_hog, y_test)
print("orb")
find_accuracy(t_orb, y_train, v_orb, y_test)
print("edges")
find_accuracy(t_edges, y_train, v_edges, y_test)
print("contours")
find_accuracy(t_contours, y_train, v_contours, y_test)
