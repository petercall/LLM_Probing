import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib import pyplot as plt
from tqdm import tqdm
from vectorize_data import vectorize_data

DATA_FILENAME = "../data/random_data.csv"
COL = "data"

MIN_DF = 2
MAX_DF = .7
NGRAM = 1
NUM_SING_VECTS = 150

CLUSTERS = [8]
KMEANS = False               #If True do KMeans, if False do Spectral Clustering (note: you can't do Silhouette visualization if you do Spectral Clustering)

SILHOUETTES = False          #This can only be True if KMeans = True
SAVE_SILHOUETTES = True    #If False it displays the silhouette plots isntead
SILHOUETTE_FILENAME = "../outputs/Random_Question_Silhouette"   #It will save the files as: SILHOUETTE_FILENAME + "_" + str(num_clusters) + ".png"

SAVE_TEXT = False           #If False it outputs the text to the terminal instead
TEXT_FILENAME = "../outputs/Random_Question_Cluster_Results.txt"

SAVE_NPY = False            #If True, this will save the data as a .npy file
SAVE_NPY_FILENAME = "../npy/Random_Question_Data"   #If SAVE_NPY = True, this will save each cluster as a .npy file named: SAVE_NPY_FILENAME + "_" + str(num_clusters) + ".npy"
SAVE_CSV = True            #iF True, this will save the labels as a new column. The column name for each cluster will be: {num_clusters}_clusters
SAVE_CSV_FILENAME = "../data/random_data.csv"   #It saves the columns as: "{num_clusters}_clusters"




#Download the data and define what your data is
data_df = pd.read_csv(DATA_FILENAME)
data = data_df[COL]

#Vectorize the data
X = vectorize_data(data, ngram = NGRAM, max_df = MAX_DF, min_df = MIN_DF, num_sing_vectors=NUM_SING_VECTS)

#Define the kmeans clustering algorithm
for num_clusters in CLUSTERS:
    if KMEANS:
        cluster = KMeans(
            n_clusters = num_clusters, 
            n_init = 10,
        )
    else:
        cluster = SpectralClustering(
            n_clusters = num_clusters,
            assign_labels="cluster_qr"
        )

    #Apply the clustering algorithm
    labels = cluster.fit_predict(X)

    #Run the silhouette score visualizer if silhouette = True
    if SILHOUETTES:
        visual = SilhouetteVisualizer(cluster, is_fitted = True)
        visual.fit(X)
        
        if SAVE_SILHOUETTES:
            save_name = SILHOUETTE_FILENAME + "_" + str(num_clusters) + ".png"
            plt.title(f"Silhouette Scores for {num_clusters} clusters")
            plt.savefig(save_name)
            plt.clf()
        else:
            visual.show()
            plt.clf()
    
    # Number of elements in each cluster
    unique, num_els = np.unique(labels, return_counts = True)
    num_els = np.sort(num_els)[::-1]
    percentages = np.round( (num_els / np.sum(num_els)) * 100, 2)

    #Output the summary statistics
    text_list = []
    text_list.append(f"Number of Clusters: {num_clusters}\n")
    text_list.append(f"Number of Singular Vectors: {NUM_SING_VECTS}\n")
    text_list.append(f"\tPercentage of elements in clusters: {percentages}\n")
    if KMEANS:
        text_list.append(f"\tClustering inertia: {round(cluster.inertia_, 2)}\n")
    if SILHOUETTES:
        text_list.append(f"\tAverage silhouette score: {round(visual.silhouette_score_, 4)}\n\n") 
    else:
        text_list.append("\n")
    
    if SAVE_TEXT:
        with open(TEXT_FILENAME, "a") as my_file:
            my_file.writelines(text_list)
    else:
        print("".join(text_list))
    
    if SAVE_NPY:
        save_name = SAVE_NPY_FILENAME + "_" + str(num_clusters) + ".npy"
        np.save(save_name, labels)
    elif SAVE_CSV:
        column_name = f"{num_clusters}_clusters"
        data_df[column_name] = labels
    
#Now save the csv file if desired
if SAVE_CSV:
    data_df.to_csv(SAVE_CSV_FILENAME, index = False)
        
        
    
    
    
    
    
    


#Output the top terms in each cluster
#original_space_centroids = lsa[0].inverse_transform(cluster.cluster_centers_)
#order_centroids = original_space_centroids.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names_out()

# string6 = ""
# for j in range(i):
#     string6 += f"Cluster {j}: "
#         for ind in order_centroids[i, :10]:
#             string6 += f"{terms[ind]} "
#             string6 += "\n"
#             string6 += "\n\n"