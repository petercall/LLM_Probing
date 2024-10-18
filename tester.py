import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classify.vectorize_data import vectorize_data
from sklearn.manifold import TSNE

my_df = pd.read_csv("data/random_data.csv")
print(my_df["8_clusters"].value_counts()*100 / my_df["8_clusters"].value_counts().sum())

