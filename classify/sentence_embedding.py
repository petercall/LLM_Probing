import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from sentence_transformers import SentenceTransformer


DATA_PATH = "../data/complete_personas.csv"
COLUMN_NAME = "answer"  #It will save embeddings in the "data" folder with filename: f"{COLUMN_NAME}_embeddings.npy"



#Get the data
my_df = pd.read_csv(DATA_PATH, header = 0)
data = my_df[COLUMN_NAME]

#Define the translator
translator = str.maketrans('', '', string.punctuation + "0123456789")

#Import the model from hugging face
model = SentenceTransformer("all-MiniLM-L6-v2")

#Create a list of cleaned strings
list_of_strings = []
for i in range(data.shape[0]):
    
    #Clean the text
    current_text = data[i]
    current_text = current_text.lower()                 #Lowercase the text
    current_text = current_text.replace("-", " ")       #Replace hyphens with spaces
    current_text = current_text.translate(translator)   #Remove punctuation and numbers
    current_text = " ".join(current_text.split())       #Remove all white space
    
    #Add the current text to current list
    list_of_strings.append(current_text)         
       

#Embed the strings
data_array = model.encode(list_of_strings)

#Save the embeddings
filename = f"../npy/{COLUMN_NAME}_embeddings.npy"
np.save(filename, data_array)