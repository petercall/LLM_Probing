import pandas as pd
import string
import numpy as np
from tqdm import tqdm

# pd.set_option("display.max_rows", 26131)

#Download the data
complete_personas = pd.read_csv("complete_personas.csv", dtype = "object")

#Define the translator and the maximum number of words
translator = str.maketrans('', '', string.punctuation)

#Define the "clean" function
def clean(text):
    #Make the string lower case
    text = text.lower()
    
    #Replace any "-" with " " to prevent the next line (which deletes punctuation) from combining hyphenated words
    text = text.replace("-", " ")

    #Delete any punctuation from the string
    text = text.translate(translator)
    
    val = text.find("\n")
    if val != -1:
        text = text[:val+1]
    
    #Make the text at most max_length number of words
    text_list = text.split()
    list_length = len(text_list)
    max_length = 10
    
    if list_length < max_length:
        max_length = list_length
    
    text_list = text_list[:max_length]
    
    return " ".join(text_list)

#Apply the "clean" function
complete_personas["classification"] = complete_personas["classification"].apply(clean)

#Save the dataset
complete_personas.to_csv("complete_personas.csv", index = False)







# #Create the empty list and loop through to get all the words in the classification list
# words = set()
# for i in range(complete_personas.shape[0]):
#     words.update(complete_personas.loc[i, "classification"].split(" "))
        
# #Create the dictionary of words
# word_dict = dict()
# for i, word in enumerate(words):
#     word_dict[word] = i
 
# #Create the numpy array
# data = np.zeros((complete_personas.shape[0], len(words)), dtype = np.int64)
# for i in tqdm(range(data.shape[0])):
#     for word in complete_personas.loc[i, "classification"].split(" "):
#         data[i,word_dict[word]] = 1