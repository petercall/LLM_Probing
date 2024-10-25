import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classify.vectorize_data import vectorize_data
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


#Load in the data
# labels = np.load("npy/answer_embeddings_8.npy")
# df = pd.read_csv("../data/complete_personas.csv")
# df["answer_embeddings_8"] = labels
# df.to_csv("data/complete_personas.csv", index=False)


