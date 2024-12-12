import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string

FILENAME = "../data/random_data3.csv"
CLUSTER_COL = "data_supervised_labels"
COL1 = "data"
COL2 = "classification"
STOPWORDS_TO_ADD = ["microsoft"]                   #A list of stopwords to add, or None
BIGRAM = True
MIN_FONT = 13


#Load in the data and specify your CLUSTER_COLumn of interest
complete_personas = pd.read_csv(FILENAME)

#Print the cluster percentages
print("Cluster percentages are as follows:")
print((complete_personas[CLUSTER_COL].value_counts() / complete_personas[CLUSTER_COL].value_counts().sum()) *100)

#Get the cluster values in the order of cluster size
cluster_vals = complete_personas[CLUSTER_COL].value_counts().index

#Add the word "help" to your stop words
if STOPWORDS_TO_ADD is not None:
    for word in STOPWORDS_TO_ADD:
        STOPWORDS.add(word)

# Print out a word cloud for each cluster
for val in cluster_vals:
    #Load in the data for that cluster
    cols = [COL1]
    if COL2 is not None:
        cols.append(COL2)
    answers = complete_personas[cols][complete_personas[CLUSTER_COL] == val].reset_index(drop = True)
    text1 = " ".join([text for text in answers[COL1]])
    if COL2 is not None:
        text2 = " ".join([text for text in answers[COL2]])
    
    #make the text lowercase, replace - with a space, and remove any new line characters
    text1 = text1.lower()
    text1 = text1.replace("-", " ")
    text1 = text1.replace("\n", " ")
    if COL2 is not None:
        text2 = text2.lower()
        text2 = text2.replace("-", " ")
        text2 = text2.replace("\n", " ")
    
    #If either of the columns are the "answer" column, perform more cleaning
    if COL1 == "answer":
        text1 = text1.replace("      ", " ")
        text1 = text1.replace("     ", " ")
        text1 = text1.replace("    ", " ")
        text1 = text1.replace("   ", " ")
        text1 = text1.replace("  ", " ")
    if COL2 == "answer":
        text2 = text2.replace("      ", " ")
        text2 = text2.replace("     ", " ")
        text2 = text2.replace("    ", " ")
        text2 = text2.replace("   ", " ")
        text2 = text2.replace("  ", " ")
    
    #Remove punctuation and also any numbers if the column is "answer"
    if COL1 == "answer":
        chars_to_remove = string.punctuation + "0123456789"    
    else:
        chars_to_remove = string.punctuation
    translator1 = str.maketrans("","",chars_to_remove)
    text1 = text1.translate(translator1)
    
    if COL2 is not None:
        if COL2 == "answer":
            chars_to_remove = string.punctuation + "0123456789"    
        else:
            chars_to_remove = string.punctuation
        translator2 = str.maketrans("","",chars_to_remove)
        text2 = text2.translate(translator2)

    #Create the word clouds
    if COL2 is not None:
        plt.subplot(121)
    wc = WordCloud(background_color = "white", colormap = "tab10", relative_scaling = 1, min_font_size = MIN_FONT, width = 800, height = 400, stopwords=STOPWORDS, collocations=BIGRAM) 
    wc.generate(text1)
    plt.imshow(wc)
    plt.title(f"Cluster: {val}")
    plt.axis("off")

    if COL2 is not None:
        plt.subplot(122)
        wc = WordCloud(background_color = "white", colormap = "tab10", relative_scaling = 1, min_font_size = MIN_FONT, width = 800, height = 400, stopwords=STOPWORDS, collocations = BIGRAM)
        wc.generate(text2)
        plt.imshow(wc)
        plt.axis("off")    
        plt.title(COL2)
        plt.suptitle(f"Wordclouds for Cluster {val}")
    
    plt.show()