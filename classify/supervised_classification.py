from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from collections.abc import Sequence
from transformers import pipeline
import numpy as np
from tqdm import tqdm

#Hyperparameters
DATA_FILE = "../data/random_data3.csv"
COLS_OF_INT = ["data"]
TARGET_CLASSES = ["science", "business", "literature", "education", "sports", "history", "art", "data analysis"]
MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 64
SAVE = True         #Will save as part of the .csv file you load in, under the column: f"{COL_OF_INTEREST}_supervised_labels"



#Download the model and tokenizer and input them into a pipeline
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = pipeline("zero-shot-classification", model = model, tokenizer = tokenizer, device = "cuda")

#Download the data
data = pd.read_csv(DATA_FILE, header = 0)

#Create the dataset class
class MyData(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


for COL_OF_INT in COLS_OF_INT:
    series = data[COL_OF_INT]

    #Create the column name where it will be saved
    col_name = f"{COL_OF_INT}_supervised_labels"
    data[col_name] = ""

    #Load it into the dataset
    dataset = MyData(series)

    #Iterate over your dataset and fill the labels Series with the model output
    try:
        for i, output in enumerate(tqdm(pipe(dataset, TARGET_CLASSES, batch_size = BATCH_SIZE), total = len(series))):
            data.loc[i, col_name] = output["labels"][0]

            if SAVE and (i % 300 == 0):
                data.to_csv(DATA_FILE, index = False)
    except:
        print("exception found")
    finally:
        #Save the output if desired
        if SAVE:
            data.to_csv(DATA_FILE, index = False)