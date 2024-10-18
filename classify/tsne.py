import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from vectorize_data import vectorize_data
from tqdm import tqdm

#Columns of interest and number of singular vectors
COL = "data"
FILENAME = "../data/random_data.csv"

TSNE_LR = [100,400,900]
TSNE_PERPLEXITY = [100,150,250]

NUM_SING_VECTS = 150
MIN_DF = 2
MAX_DF = .7
NGRAM = 2

SAVE = True     #If SAVE = False then it will display the plots instead





#Get the combined TSNE arguments
tsne_args = [(a,b) for a in TSNE_LR for b in TSNE_PERPLEXITY]
print(tsne_args)

#Download the data
data = pd.read_csv(FILENAME, dtype = "object")
X1 = data[COL]
X1 = vectorize_data(X1, ngram = NGRAM, max_df = MAX_DF, min_df = MIN_DF, num_sing_vectors=NUM_SING_VECTS)

for lr, perp in tqdm(tsne_args):
    # #Create the TSNE object and run it on the data
    tsne = TSNE(perplexity = perp, learning_rate=lr)
    X1_new = tsne.fit_transform(X1)
    
    # Plot the original t-SNE data
    plt.scatter(X1_new[:,0], X1_new[:,1], marker = '.', s = 10)
    plt.title(f"t-SNE, Perplexity = {perp}, LR = {lr}")
    plt.axis("off")
    
    if SAVE:
        file = f"tsne_{COL}_Perp{perp}_LR{lr}.png"
        plt.savefig(f"../outputs/{file}")         
    else:
        plt.show()
    plt.clf()














#Plot the data, color-coded according to its KMeans cluster
# colors = ["blue", "green", "orange", "red", "yellow", "black", "brown", "cyan", "purple", "pink", "olive"]
# for i in range(8):
#     mask = (data[:,2] == i)
#     new_data = data[mask]
#     plt.scatter(new_data[:,0], new_data[:,1], color = colors[i])
# plt.title("t-SNE KMeans Clustering")
# plt.axis("off")
# plt.show()