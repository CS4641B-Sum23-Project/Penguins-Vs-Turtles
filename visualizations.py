import os
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CLASS_MAP = {1 : 'Penguin', 2 : 'Turtle'}


def plot_PCA(features : np.ndarray, _class : np.ndarray, title : str, save_img_path : str = None,
             show_plot : bool = False) -> np.ndarray:
  # perform PCA on the features
  
  standard_scaler = StandardScaler()
  scaled_features = standard_scaler.fit_transform(features)
  
  pca = PCA(n_components=2)
  pca_results = pca.fit_transform(scaled_features)
  
  if save_img_path or show_plot:
    plt.figure(figsize=(10,6))
    class_1 = _class == 1
    plt.scatter(pca_results[class_1,0], pca_results[class_1,1], label=CLASS_MAP[1], color='red', s=50, alpha=0.7 )
    class_2 = _class == 2
    plt.scatter(pca_results[class_2,0], pca_results[class_2,1], label=CLASS_MAP[2], color='blue', s=50, alpha=0.7 )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title(title)
  if save_img_path:
    plt.savefig(save_img_path)
  if show_plot:
    plt.show()
  
  return pca_results

def plot_Kernel_PCA(features : np.ndarray, _class : np.ndarray, title : str, save_img_path : str = None,
                    show_plot : bool = False) -> np.ndarray:
  KPCA = KernelPCA(n_components=2, kernel='rbf', gamma=None)
  kpca_results = KPCA.fit_transform(features)
  if save_img_path or show_plot:
    plt.figure(figsize=(10,6))
    class_1 = _class == 1
    plt.scatter(kpca_results[class_1,0], kpca_results[class_1,1], label=CLASS_MAP[1], color='red', s=50, alpha=0.7 )
    class_2 = _class == 2
    plt.scatter(kpca_results[class_2,0], kpca_results[class_2,1], label=CLASS_MAP[2], color='blue', s=50, alpha=0.7 )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title(title)
  if save_img_path:
    plt.savefig(save_img_path)
  if show_plot:
    plt.show()
  
  return kpca_results
def kmeans_centroids(training_df : pd.DataFrame, validation_df : pd.DataFrame, 
                     save_img_path : str = None, show_plot : bool = False) -> None:
   # get data and store in numpy array
  data = np.array(training_df['64x64'].apply(lambda x : x.flatten()).tolist())
  
  # normalize
  data = data/255.0

  kmeans = KMeans(n_clusters=2, random_state=0)
  cluster = kmeans.fit_predict(data)
  shape = kmeans.cluster_centers_.shape[1]
  

  x_data = np.arange(shape).tolist()
  plt.scatter(x_data,kmeans.cluster_centers_[0], color='red', alpha=0.2, s=70)
  plt.scatter(x_data,kmeans.cluster_centers_[1], color='blue', alpha=0.2, s=50)
  if save_img_path:
    plt.savefig(save_img_path)
  if show_plot:
    plt.show()

  
def generate_visuals(training_df : pd.DataFrame, validation_df : pd.DataFrame, features_dict : dict) -> None:
  plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
  if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
  
  bb_features = np.array(features_dict['bb_Mobilenet'].tolist())
  non_bb_features = np.array(features_dict['Mobilenet'].tolist())
  HOG = np.array(features_dict['HOG'].apply(lambda x : x.flatten()).tolist())
  ORB = np.array(features_dict['ORB'].apply(lambda x : x.flatten()).tolist())
  edges = np.array(features_dict['edges'].apply(lambda x : x.flatten()).tolist())
  # contours = np.array(features_dict['contours'].apply(lambda x : x.flatten()).tolist())
  
  results = {}
  path = os.path.join(plots_dir, 'Canny_Edges_Features_KPCA.jpg')
  _result = \
    plot_Kernel_PCA(edges, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of Canny Edges", save_img_path=path)
  
  results['edge_kpca'] = _result
  # path = os.path.join(plots_dir, 'Contours_Features_KPCA.jpg')
  # plot_Kernel_PCA(contours, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of Contours", save_img_path=path)
  
  path = os.path.join(plots_dir, 'ORB_Features_KPCA.jpg')
  _result = \
    plot_Kernel_PCA(ORB, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of ORB Features", save_img_path=path)
  results['orb_kpca'] = _result
  
  path = os.path.join(plots_dir, 'HOG_Features_KPCA.jpg')
  _result = \
    plot_Kernel_PCA(HOG, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of HOG Translation", save_img_path=path)
  results['hog_kpca'] = _result
  
  path = os.path.join(plots_dir, 'MobileNetV2_Features_BB_KPCA.jpg')
  _result = \
    plot_Kernel_PCA(bb_features, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of MobileNetV2 Bounding Box Features", save_img_path=path)
  results['mobilenet_bb_kpca'] = _result
  
  path = os.path.join(plots_dir, 'MobileNetV2_Features_KPCA.jpg')
  _result = \
    plot_Kernel_PCA(non_bb_features, training_df.category_id.to_numpy(), "Kernel_PCA Visualization of MobileNetV2 Non-Bounding Box Features", save_img_path=path)
  results['mobilenet_kpca'] = _result
 
 
 
 
 
  path = os.path.join(plots_dir, 'Canny_Edges_Features_PCA.jpg')
  _result = \
    plot_PCA(edges, training_df.category_id.to_numpy(), "PCA Visualization of Canny Edges", save_img_path=path)
  results['edge_pca'] = _result
  
  # path = os.path.join(plots_dir, 'Contours_Features_PCA.jpg')
  # plot_Kernel_PCA(contours, training_df.category_id.to_numpy(), "PCA Visualization of Contours", save_img_path=path)
  
  path = os.path.join(plots_dir, 'ORB_Features_PCA.jpg')
  _result = \
    plot_PCA(ORB, training_df.category_id.to_numpy(), "PCA Visualization of ORB Features", save_img_path=path)
  results['orb_pca'] = _result
  
  path = os.path.join(plots_dir, 'HOG_Features_PCA.jpg')
  _result = \
    plot_PCA(HOG, training_df.category_id.to_numpy(), "PCA Visualization of HOG Translation", save_img_path=path)
  results['hog_pca'] = _result
  
  path = os.path.join(plots_dir, 'MobileNetV2_Features_BB_PCA.jpg')
  _result = \
    plot_PCA(bb_features, training_df.category_id.to_numpy(), "PCA Visualization of MobileNetV2 Bounding Box Features", save_img_path=path)
  results['mobilenet_bb_pca'] = _result
  
  path = os.path.join(plots_dir, 'MobileNetV2_Features_PCA.jpg')
  _result = \
    plot_PCA(non_bb_features, training_df.category_id.to_numpy(), "PCA Visualization of MobileNetV2 Non-Bounding Box Features", save_img_path=path)
  results['mobilenet_pca'] = _result
  
  with open('data/results.pkl', 'wb') as f:
    import pickle
    pickle.dump(results, f)
  
  
