import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List
import pickle
import pdb
def make_plots() -> None:
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  output_dir = os.path.join(cur_dir, 'plots', 'DecisionTrees')
  os.makedirs(output_dir, exist_ok=True)
  RF_pickles = glob.glob('data/Random_Forest*.pkl')
  DT_pickles = glob.glob('data/Decision_Tree*.pkl')
  
  RF_data = {}
  DT_data = {}
  for path in RF_pickles:
    name = os.path.basename(path).split('.')[0].split('Forest_')[1]
    with open(path, 'rb') as f:
      data = pickle.load(f)
    RF_data[name] = data
  for path in DT_pickles:
    name = os.path.basename(path).split('.')[0].split('Tree_')[1]
    with open(path, 'rb') as f:
      data = pickle.load(f)
    DT_data[name] = data


  colors = ['green', 'blue', 'orange', 'red', 'purple', 'cyan', 'magenta']
  legends = ['bb', 'non-bb', 'grayscale', 'HOG', 'ORB', 'contours', 'edges']
  data_names = ['mobile_net_bounding_box', 'mobile_net_no_bounding_box', 'grayscale', 
              'hog_filters', 'orb_features', 'contours', 'canny_edges']
  DT : List[pd.DataFrame] = [DT_data[name] for name in data_names]
  RF : List[pd.DataFrame] = [RF_data[name] for name in data_names]
  # All features
  plt.figure(figsize=(14,10))
  for color, legend, dt_df, rf_df in zip(colors, legends, DT, RF):
    dt_means = dt_df.groupby('max_depth')['accuracy'].mean()
    x_index = dt_means.index.tolist()
    dt_data = dt_means.tolist()
    rf_data = rf_df.groupby('max_depth')['accuracy'].mean().tolist()
    plt.plot(x_index, dt_data, color=color, linestyle='dashed', label=f'DT-{legend}')
    plt.plot(x_index, rf_data, color=color, label=f'RF-{legend}')

  plt.xticks(x_index)
  plt.title('All Acurracy Vs Maximum Depth', fontsize=20)
  plt.ylabel('Average Accuracy', fontsize=16)
  plt.xlabel('Maximum Depth', fontsize=16)
  plt.legend(loc='upper left', bbox_to_anchor=(.95, 1.0), fontsize='large')
  outputpath = os.path.join(output_dir, 'max_depth_over_all_accuracy.jpg')
  plt.savefig(outputpath)
  plt.clf()
  
  # Just mobile net features, because they performed the best
  colors = ['green', 'blue']
  legends = ['bb', 'non-bb', ]
  data_names = ['mobile_net_bounding_box', 'mobile_net_no_bounding_box']
  for color, legend, dt_df, rf_df in zip(colors, legends, DT, RF):
    dt_means = dt_df.groupby('max_depth')['accuracy'].mean()
    x_index = dt_means.index.tolist()
    dt_data = dt_means.tolist()
    rf_data = rf_df.groupby('max_depth')['accuracy'].mean().tolist()
    plt.plot(x_index, dt_data, color=color, linestyle='dashed', label=f'DT-{legend}')
    plt.plot(x_index, rf_data, color=color, label=f'RF-{legend}')
    plt.xticks(x_index)
  plt.title('MobileNet Acurracy Vs Maximum Depth', fontsize=20)
  plt.ylabel('Average Accuracy', fontsize=16)
  plt.xlabel('Maximum Depth', fontsize=16)
  plt.legend(loc='upper left', bbox_to_anchor=(.95, 1.0), fontsize='large')
  outputpath = os.path.join(output_dir, 'max_depth_MobileNet_accuracy.jpg')
  plt.savefig(outputpath)
  
  plt.clf()
  
  for color, legend, dt_df, rf_df in zip(colors, legends, DT, RF):
    dt_means = dt_df.groupby('max_depth')['precision'].mean()
    x_index = dt_means.index.tolist()
    dt_data = dt_means.tolist()
    rf_data = rf_df.groupby('max_depth')['precision'].mean().tolist()
    plt.plot(x_index, dt_data, color=color, linestyle='dashed', label=f'DT-{legend}')
    plt.plot(x_index, rf_data, color=color, label=f'RF-{legend}')
    plt.xticks(x_index)
  plt.title('MobileNet Precision Vs Maximum Depth', fontsize=20)
  plt.ylabel('Average Precision', fontsize=16)
  plt.xlabel('Maximum Depth', fontsize=16)
  plt.legend(loc='upper left', bbox_to_anchor=(.95, 1.0), fontsize='large')
  outputpath = os.path.join(output_dir, 'max_depth_MobileNet_Precision.jpg')
  plt.savefig(outputpath)
  plt.clf()


  DT_df = DT_data['mobile_net_bounding_box'] 
  ACC = DT_df['accuracy']
  DEPTH = DT_df['max_depth']
  FEATURES = DT_df['max_features']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(DEPTH, FEATURES, ACC)
  ax.set_xlabel('Maximum Depth')
  ax.set_ylabel('Maximum Features')
  ax.set_zlabel('Accuracy')
  
  ax.set_title('Accuracy from Maximum Depth and Maximum Features')
  outputpath = os.path.join(output_dir, 'MN_BB_Accuracy_vs_Depth_and_Features.jpg')
  plt.savefig(outputpath)
  plt.clf()

  depth = 6
  DT6 = DT_df[DT_df['max_depth'] == depth]
  x = DT6['max_features'].tolist()
  y = DT6['accuracy'].tolist()
  plt.title("MobileNet BB, DT, Max_Depth=6")
  plt.xlabel("Max Features", fontsize=16)
  plt.ylabel("Accuracy", fontsize=16)
  plt.plot(x,y)
  outputpath = os.path.join(output_dir, 'max_depth_6_DT_accuracy.jpg')
  plt.savefig(outputpath)
  plt.clf()
  
  
  DF_df = RF_data['mobile_net_bounding_box'] 
  depth = 11
  DF11 = DF_df[DF_df['max_depth'] == depth]
  X = DF11['estimators']
  Y = DF11['max_features']
  Z = DF11['accuracy']

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(X, Y, Z)
  ax.set_xlabel('Estimators')
  ax.set_ylabel('Maximum Features')
  ax.set_zlabel('Accuracy')
  ax.set_title('Accuracy from # Estimators and Maximum Features')
  outputpath = os.path.join(output_dir, 'MN_BB_Accuracy_vs_Estimators_and_Features.jpg')
  plt.savefig(outputpath)
  plt.clf()
  
  max_features = 0.8
  DF11P8 = DF11[(DF11['max_features'] > 0.79) & (DF11['max_features'] < 0.81)]
  estimators = DF11P8['estimators']
  acc = DF11P8['accuracy']
  
  plt.plot(estimators, acc)
  plt.title("MobileNet BB, RF, Max_Depth=11, Max_Features=80%")
  plt.xlabel("Estimators")
  plt.ylabel("Accuracy")
  outputpath = os.path.join(output_dir, 'MN_BB_Accuracy_vs_Estimators_MD=11_MF=80%.jpg')
  plt.savefig(outputpath)
  
  pdb.set_trace()
  # Considering DT did better lets plot on the max_depth that did the best
if __name__ == '__main__':
  make_plots()