from typing import Tuple
import json
import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse as ap
import numpy as np
import img_utls
from main import generate_data, load_data
import feature_extractions as fe

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

def convert_features(features) -> Tuple[np.ndarray, np.ndarray]:
    # Existing code to load the features dictionary
    feature_dict = features
    
    # Extract the features from the dictionary
    bb_features = feature_dict['bb_Mobilenet']
    mobilenet_features = feature_dict['Mobilenet']
    hog_features = feature_dict['HOG']
    orb_features = feature_dict['ORB']
    edges_features = feature_dict['edges']
    contours_features = feature_dict['contours']

    # print(type(bb_features))
    # print(mobilenet_features)
    # print(hog_features)
    # print(orb_features)
    # print(edges_features)
    # print(contours_features)

    # Run for bb_features and mobilenet_features
    X_train = pd.DataFrame(bb_features.tolist()).values

    # Run for hog_features, orb_features, edges_features
    # X_train_3d = np.array(hog_features.to_list())
    # X_train = X_train_3d.reshape(X_train_3d.shape[0], -1)

    # TODO finsih 
    # Run for contour_features
    # X_train = pd.Series([np.concatenate(item) if item else np.zeros((1, 2)) for item in contours_features if item])
    

    return X_train
# Split the dataset into training and testing sets
X = convert_features(features)

# Get labels from train_annotations
with open("train_annotations",'r') as file:
    data = json.load(file)
y = [obj["category_id"] for obj in data]
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