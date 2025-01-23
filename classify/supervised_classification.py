from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from collections.abc import Sequence
from transformers import pipeline
import numpy as np
from tqdm import tqdm

#Hyperparameters
DATA_FILE = "../data/datasets/tiny_stories_subset_20000.csv"
SUBSET_SIZE = None                    #If None it will use the entire dataset, if a number it will take a random subset of the dataset of this size
COLS_OF_INT = ["text"]
TARGET_CLASSES = ["science", "mathematics", "business", "literature", "education", "sports", "history", "art", "computer programming", "law", "medicine"]
MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 64
PARTIAL_SCORES = True   #If this is True, it will sum the scores for each data point rather than just saving the top one
SAVE = False            #Will save as part of the .csv file you load in, under the column: f"{COL_OF_INTEREST}_supervised_labels".
                        #If SUBSET_SIZE is not None, it will save as a new .csv file with name: f"{DATA_FILE}_subset_{SUBSET_SIZE}.csv"
                        #If False it will print to the console

#Download the model and tokenizer and input them into a pipeline
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = pipeline("zero-shot-classification", model = model, tokenizer = tokenizer, device = "cuda")

#Download the data
data = pd.read_csv(DATA_FILE, header = 0)

#Get a subset of the data if desired
if SUBSET_SIZE is not None:
    data = data.sample(SUBSET_SIZE).reset_index(drop = True)

#Create a final_scores variable if PARTIAL_SCORES is True
if PARTIAL_SCORES:
    final_scores = np.zeros(len(TARGET_CLASSES))
    final_i = 0

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

            # if SAVE and (i % 300 == 0):
            #     if SUBSET_SIZE is not None:
            #         data.to_csv(f"{DATA_FILE[:-4]}_subset_{SUBSET_SIZE}.csv", index = False)
            #     else:
            #         data.to_csv(DATA_FILE, index = False)                
            
            if PARTIAL_SCORES:
                labels = output["labels"]
                scores = output["scores"]
                
                order = np.argsort(labels)
        
                sorted_labels = np.array(labels)[order]
                sorted_scores = np.array(scores)[order]
                
                final_scores += sorted_scores
                final_i += 1  
        
    except:
        print("exception found")
    finally:
        #Save the output if desired
        if SAVE:
            if SUBSET_SIZE is not None:
                data.to_csv(f"{DATA_FILE[:-4]}_subset_{SUBSET_SIZE}.csv", index = False)
            else:
                data.to_csv(DATA_FILE, index = False)             
            
        
        if PARTIAL_SCORES:
            print(f"Label Order: {np.sort(TARGET_CLASSES)}")
            # print(f"Final Scores: {final_scores}")
            print(f"Final Proportions: {final_scores/final_i}")