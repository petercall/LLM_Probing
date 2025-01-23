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
# df = pd.read_csv("data/datasets/tiny_stories_subset_20000.csv", header = 0)
# print(df)]


AMOUNT_OF_DATA = 10
my_series = pd.Series([""]*AMOUNT_OF_DATA, dtype = "object", name = "answer")
print(my_series)