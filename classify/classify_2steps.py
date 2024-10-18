import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from vectorize_data import vectorize_data


def cluster_centroids(centroids, num_clusters = 8, do_kmeans = False):
    """
    This code uses Spectral Clustering to cluster n centroids that were originally found via KMeans on a larger dataset.
        
    Inputs:
        centroids: 2D Numpy array (shape = nxd where n = # of centroids, d = dimension of each centroid)
            
        num_clusters: int (default = 8)
            This is the number of clusters to cluster the centroids into.
            
        do_kmeans: bool (default = False)
            By default, the cluster_qr method will be used to cluster the top k eigenvectors of the graph laplacian of the affinity matrix.
            If you want it to perform KMeans as well, you can specify this by setting do_kmeans = True.
        
    
    Returns:
        result: 1D Numpy array (length = n) or 2D Numpy array (shape: nx2)
            If do_kmeans = False, this is a 1D array of size n = # of centroids, where element i corresponds to the cluster that the ith centroid belongs to.
            If do_kmeans = True, this is a 2D array of shape nx2, where the first column is the clustering found using cluster_qr, and the 2nd is the clustering found using KMeans
    """

    #Cluster the data
    cluster1 = SpectralClustering(n_clusters = num_clusters, assign_labels = "cluster_qr")
    cluster1.fit(centroids)

    if do_kmeans:
        cluster2 = SpectralClustering(n_clusters = num_clusters, assign_labels = "kmeans")
        cluster2.fit(centroids)

    #Return either a 1D or 2D array, depending on if do_kmeans False or do_kmeans = True
    if do_kmeans:
        return np.column_stack([cluster1.labels_, cluster2.labels_])
    else:
        return cluster1.labels_


def label_original(original_labels, centroid_labels):
    """
    This code assigns each data point the cluster label that the centroid it is attached to was assigned during spectral clustering
        
    Input:
        original_labels: 1D Numpy array (length = # of original data points)
            Each element of this array corresponds to the centroid that the original data point was assigned in the first clustering
            
        centroid_labels: 1D Numpy array (length = # of centroids)
            Each element of this array corresponds to the cluster that the centroid was put into in the second clustering
            
    Returns:
        new_labels: 1D Numpy array (length = # or original data points)
            Each element of this array gives the final cluster that the data point belongs to after the second clustering
    """

    #Create the empty new_labels array
    new_labels = np.zeros(original_labels.shape[0])

    #Populate new_labels with the label that the associated cluster centroid has
    for i in range(original_labels.shape[0]):
        new_labels[i] = centroid_labels[original_labels[i]]
    
    #Return new_labels
    return new_labels
    

COL = "persona"
NUM_SING_VECTS = 150
MIN_DF = 2
MAX_DF = .7
NGRAM = 1
N_CLUSTERS_FIRST = 30000
N_CLUSTERS_SECOND = 8
DO_KMEANS = False




#Download the data and define what your data is
complete_personas = pd.read_csv("../data/complete_personas.csv", dtype = "object")
data = complete_personas[COL]

#Vectorize the data
X = vectorize_data(data, ngram = NGRAM, max_df = MAX_DF, min_df = MIN_DF, num_sing_vectors=NUM_SING_VECTS)

# Define the kmeans clustering algorithm
cluster = KMeans(
    n_clusters = N_CLUSTERS_FIRST, 
    n_init = 10,
)

#Apply the clustering algorithm
labels = cluster.fit_predict(X)

#Cluster the centroids
centroid_labels = cluster_centroids(cluster.cluster_centers_ ,num_clusters = N_CLUSTERS_SECOND, do_kmeans = DO_KMEANS)
    
#Get the labels of the original data
final_labels = label_original(labels, centroid_labels)

#Save the final labels
np.save("personas_labels.npy", final_labels)