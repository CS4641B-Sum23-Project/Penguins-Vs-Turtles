import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score, fowlkes_mallows_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import sklearn.metrics
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.model_selection import RandomizedSearchCV

Word_Mapping = {
  'edge_kpca': 'Canny Edges Kernel PCA', 
  'orb_kpca': 'ORB Kernel PCA', 
  'hog_kpca': 'HOG Kernel PCA', 
  'mobilenet_bb_kpca': 'MobileNetV2 No Bounding Box Kernel PCA', 
  'mobilenet_kpca': 'MobileNetV2 Bounding Box Kernel PCA', 
  'edge_pca': 'Canny Edges PCA',
  'orb_pca': 'ORB PCA',
  'hog_pca': 'HOG PCA',
  'mobilenet_bb_pca': 'MobileNetV2 No Bounding Box PCA',
  'mobilenet_pca': 'MobileNetV2 Bounding Box PCA'
}
def process_data():
  with open('data_final/results.pkl', 'rb') as f:
    results = pickle.load(f)
  
  with open('data_final/image_data.pkl', 'rb') as f:
    image_data = pickle.load(f)
  
  truth = image_data['training_df'].category_id.to_numpy()
  truthA = truth - 1
  truthB = 1 - truthA
  cluster_results = {}
  
  results_keys = results.keys()
  
  # Use KMeans
  N_Clusters = 2
  kmeans = KMeans(n_clusters=N_Clusters, random_state=np.random.randint(0, 10000000))
  kmeans_results = {}
  for key in results_keys:
    
    print(key)
    data = results[key]

    kmeans.fit(data)
    labels = kmeans.labels_
    C1 = data[labels == 0]
    C2 = data[labels == 1]
    
    plt.scatter(C1[:, 0], C1[:,1], label='Cluster 1', color='red', alpha=.7)
    plt.scatter(C2[:, 0], C2[:,1], label='Cluster 2', color='blue', alpha=.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], marker='*', color='green', s=150, label='Cluster Centers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering on ' + Word_Mapping[key])
    plt.legend()
    # plt.show()
    f_name = 'plots/cluster_kmean_'+key + '.jpg'
    
    nmi = max(normalized_mutual_info_score(truthA, labels), normalized_mutual_info_score(truthB, labels))
    folkes = max(fowlkes_mallows_score(truthA, labels), fowlkes_mallows_score(truthB, labels))
    dbi = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels)
    # print(f"NMI = {nmi}")
    # print(f"Folkes Mallow = {folkes}")
    # print(f"DBI = {dbi}")
    # print(f"Sil Coef = {sil}")
    res = {'NMI':nmi, 'Folkes Mallow':folkes, 'DBI':dbi, 'Silhoutte':sil}

    
    kmeans_results[key] = res
    plt.savefig(f_name)
    plt.clf()
    f_name = 'plots/cluster_kmean_'+key+'_metrics.jpg'
    x = res.keys()
    y = res.values()
    plt.bar(x, y)
    plt.ylabel('Score')
    plt.title(f"Clustering Metrics with KMeans on {Word_Mapping[key]}")
    plt.savefig(f_name)
    plt.clf()
    #pdb.set_trace()
    #print("Here")
  cluster_results['kmeans'] = kmeans_results
  
  # GMM 
  gmm = GaussianMixture(n_components=2)
  gmm_results = {}
  for key in results_keys:
    
    print(key)
    data = results[key]

    gmm.fit(data)
    labels = gmm.predict(data)
    # pdb.set_trace()
    C1 = data[labels == 0]
    C2 = data[labels == 1]
    means = gmm.means_
    cov = gmm.covariances_
    plt.scatter(C1[:, 0], C1[:,1], label='Cluster 1', color='red', alpha=.7)
    plt.scatter(C2[:, 0], C2[:,1], label='Cluster 2', color='blue', alpha=.7)
    for i in range(len(means)):
      cov_mat = cov[i]
      eigvals, eigvecs = np.linalg.eigh(cov_mat)
      angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
      width, height = 2 * np.sqrt(2 * eigvals)
      ellipse = Ellipse(xy=means[i], width=width, height=height, angle=angle,
                        edgecolor='green', facecolor='none')
      plt.gca().add_patch(ellipse)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('GMM Clustering on ' + Word_Mapping[key])
    plt.legend()
    # plt.show()
    f_name = 'plots/cluster_gmm_'+key + '.jpg'
    
    nmi = max(normalized_mutual_info_score(truthA, labels), normalized_mutual_info_score(truthB, labels))
    folkes = max(fowlkes_mallows_score(truthA, labels), fowlkes_mallows_score(truthB, labels))
    dbi = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels)
    print(f"NMI = {nmi}")
    print(f"Folkes Mallow = {folkes}")
    print(f"DBI = {dbi}")
    print(f"Sil Coef = {sil}")
    res = {'NMI':nmi, 'Folkes Mallow':folkes, 'DBI':dbi, 'Silhoutte':sil}

    
    gmm_results[key] = res
    plt.savefig(f_name)
    plt.clf()
    f_name = 'plots/cluster_gmm_'+key+'_metrics.jpg'
    x = res.keys()
    y = res.values()
    plt.bar(x, y)
    plt.ylabel('Score')
    plt.title(f"Clustering Metrics with GMM on {Word_Mapping[key]}")
    plt.savefig(f_name)
    plt.clf()
    #pdb.set_trace()
    print("Here")
  cluster_results['gmm'] = gmm_results
  
  #DBSCAN
  dbscan = DBSCAN(eps=0.1, min_samples=40)
  dbscan_results = {}
  
  for key in results_keys:
    try:
      print(key)
      data = results[key]
      # pdb.set_trace()
      # random_search = RandomizedSearchCV(estimator=dbscan, param_distributions=params,
      #                                    n_iter=15, cv=5, scoring=silhouette_scorer)
      # random_search.fit(data)
      # best_dbscan = random_search.best_estimator_
      
      dbscan.fit(data)
      labels = dbscan.labels_
      labels += 1
      pdb.set_trace()
      C1 = data[labels == 0]
      C2 = data[labels == 1]
      Outlier = data[labels == -1]

      plt.scatter(data[:, 0], data[:,1], c=labels, cmap='viridis')

      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.title('DBSCAN Clustering on ' + Word_Mapping[key])
      # plt.show()
      f_name = 'plots/cluster_dbscan_'+key + '.jpg'
      
      nmi = max(normalized_mutual_info_score(truthA, labels), normalized_mutual_info_score(truthB, labels))
      folkes = max(fowlkes_mallows_score(truthA, labels), fowlkes_mallows_score(truthB, labels))
      #dbi = davies_bouldin_score(data, labels)
      sil = silhouette_score(data, labels)
      print(f"NMI = {nmi}")
      print(f"Folkes Mallow = {folkes}")
      #print(f"DBI = {dbi}")
      print(f"Sil Coef = {sil}")
      res = {'NMI':nmi, 'Folkes Mallow':folkes, 'Silhoutte':sil}

      
      dbscan_results[key] = res
      plt.savefig(f_name)
      plt.clf()
      f_name = 'plots/cluster_dbscan_'+key+'_metrics.jpg'
      x = res.keys()
      y = res.values()
      plt.bar(x, y)
      plt.ylabel('Score')
      plt.title(f"Clustering Metrics with DBSCAN on {Word_Mapping[key]}")
      plt.savefig(f_name)
      plt.clf()
      #pdb.set_trace()
    except:
      plt.clf()
      continue
  
  
if __name__ == '__main__':
  process_data()