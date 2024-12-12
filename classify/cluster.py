import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib import pyplot as plt
from tqdm import tqdm
from old.vectorize_data import vectorize_data
from old.sentence_embedding import embed


DATA_FILENAME = "../data/random_data_gemma.csv"
COL = "answers"                                        

EMBEDDINGS = False                                  #If True, it will embed the data using the SentenceTransformer model, if False it will use PCA and TF-IDF

MIN_DF = 2                                          #These are all ignored if EMBEDDINGS = True
MAX_DF = .6
NGRAM = 1
NUM_SING_VECTS = 150

RANDOM_SUBSET = False                               #If True, it will take a random subset of the data. This can be good to test out a couple of custer sizes via Silhouette visualization (which is much faster with smaller amounts of data)
RANDOM_SUBSET_SIZE = 30000                          #If RANDOM_SUBSET = True, this is the size of the random subset. If RANDOM_SUBSET = False, this is ignored.

CLUSTERS = [8]
KMEANS = False                                       #If True do KMeans, if False do Spectral Clustering (note: you can't do Silhouette visualization if you do Spectral Clustering)

SILHOUETTES = False                                  #This can only be True if KMeans = True
SAVE_SILHOUETTES = False                             #If False it displays the silhouette plots isntead
SILHOUETTE_FILENAME = "../outputs/answer_embeddings"#Ignored if SAVE_SILHOUETTES = False. If True, it will save the files as: SILHOUETTE_FILENAME + "_" + str(num_clusters) + ".png"

SAVE_TEXT = False                                    #If False it outputs the text to the terminal instead
TEXT_FILENAME = "../outputs/answer_embedding_results.txt"

SAVE_CSV = True                                    #If True it saves the labels as a new column in the dataframe with the name: f"{COL}_labels_({num_clusters} clusters)"





#Download the data and define what your data is
data_df = pd.read_csv(DATA_FILENAME)
data = data_df[COL]

#Vectorize the data
if EMBEDDINGS:
    X = embed(data)
else:
    X = vectorize_data(data, ngram = NGRAM, max_df = MAX_DF, min_df = MIN_DF, num_sing_vectors=NUM_SING_VECTS)
    
#Take a random subset of the data if desired
if RANDOM_SUBSET:
    random_indices = np.random.choice(X.shape[0], RANDOM_SUBSET_SIZE, replace=False)
    X = X[random_indices]

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
    if not EMBEDDINGS:
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
    
    #If we are going to save the data, then append the labels to the dataframe
    if SAVE_CSV:
        column_name = f"{COL}_labels_({num_clusters} clusters)"
        data_df[column_name] = labels
    
    
#Save the csv file if specified to do so
if SAVE_CSV:
    data_df.to_csv(DATA_FILENAME, index = False)