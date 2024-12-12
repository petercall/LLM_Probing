import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm

FILE = "data/variableTemp_llama.csv"
COL = "answer"
LABEL_COL = "answer_supervised_labels"


#hyperparameters
TARGET_CLASSES = ["science", "business", "literature", "education", "sports", "history", "art", "data analysis"]
MODEL_NAME = "facebook/bart-large-mnli"

#Download the data
data = pd.read_csv(FILE, dtype = "object")
print(data[LABEL_COL].value_counts() * 100 / data.shape[0])





#Download the model and tokenizer and input them into a pipeline
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# pipe = pipeline("zero-shot-classification", model = model, tokenizer = tokenizer, device = "cuda")

# for i in tqdm(range(data.shape[0])):
#     if pd.isna(data.at[i,LABEL_COL]):
#         data.at[i, LABEL_COL] = pipe(data.at[i, COL], TARGET_CLASSES)["labels"][0]
        
# data.to_csv(FILE, index = False)
