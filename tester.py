import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
import kaggle, kagglehub
import os

#download the data
df = pd.read_csv("data/model_output/variableTemp_phi3_500_[2.75]_bos.csv", header = 0)
# print(df)
df_sample = df.sample(10)

for i, index in enumerate(df_sample.index):
    print(f"{i}: {df_sample.at[index, "0"]}")
    print()
