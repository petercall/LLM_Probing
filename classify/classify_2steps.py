import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from old.vectorize_data import vectorize_data
from old.sentence_embedding import embed


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



CSV_FILENAME = "../data/random_data2.csv"          
COL = "data" 

EMBEDDINGS = True                               #If True, it will use sentence embeddings. If False, it will use PCA and TF-IDF
                                
NUM_SING_VECTS = 150                            #These are all ignored if EMBEDDINGS = True
MIN_DF = 2
MAX_DF = .7
NGRAM = 1

N_CLUSTERS_FIRST = 30000
N_CLUSTERS_SECOND = [8]                         #The number of cluseters to do in the second clustering. If you specify several, it will do all of them
DO_KMEANS = False                               #This is whether to do KMeans on the 2nd clustering. If False, it does Spectral Clustering instead (which is more powerful)

SAVE_CSV = True                                 #If True, it saves one new column in the dataframe for each clustering, under the column header: f"{COL}_labels_(2step, {clusts} clusters)"





#Download the data and define what your data is
df = pd.read_csv(CSV_FILENAME, header = 0)
data = df[COL]

#Vectorize the data
if EMBEDDINGS:
    X = embed(data)
else:
    X = vectorize_data(data, ngram = NGRAM, max_df = MAX_DF, min_df = MIN_DF, num_sing_vectors=NUM_SING_VECTS)
    
#Allocate room for the final clustering
final_labels = np.empty(shape = (X.shape[0],len(N_CLUSTERS_SECOND)))

#_kmneas Define the kmeans clustering algorithm
cluster_kmeans = KMeans(
    n_clusters = N_CLUSTERS_FIRST, 
    n_init = 10,
)

#Apply the clustering algorithm
labels = cluster_kmeans.fit_predict(X)

#Cluster the centroids
for i, clusts in enumerate(N_CLUSTERS_SECOND):
    #Cluster the centroids using Spectral Clustering
    cluster_spectral = SpectralClustering(n_clusters = clusts, assign_labels = "cluster_qr")
    cluster_spectral.fit(cluster_kmeans.cluster_centers_)
    centroid_labels = cluster_spectral.labels_
    
    #Get the labels of the original data
    final_labels[:,i] = label_original(labels, centroid_labels)
    
    
if SAVE_CSV:
    for i, clusts in enumerate(N_CLUSTERS_SECOND):
        column_name = f"{COL}_labels_(2step, {clusts} clusters)"
        df[column_name] = final_labels[:,i]
        
    #Save the dataframe   
    df.to_csv(CSV_FILENAME, index = False)