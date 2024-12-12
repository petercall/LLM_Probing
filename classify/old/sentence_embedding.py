import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from sentence_transformers import SentenceTransformer
from collections.abc import Iterable
import scipy.linalg as la


def embed(data: Iterable[str], similarities = False) -> np.ndarray:
    """
    Embeds text data using a pre-trained SentenceTransformer model.
        - The text data is preprocessed by converting to lowercase, replacing hyphens with spaces, removing punctuation and numbers, and removing all whitespace except for a single space in between words.
        - The embeddings are generated using the "all-mpnet-base-v2" model from the SentenceTransformer library, which is the current best model in that library.
    
    Args:
        data (Iterable[str]): An iterable of strings to embed (has the __iter__ function defined so can be used in a for loop).
        
    Returns:
        np.ndarray (len(data) x 768): An array of embeddings for the text data
        similarities
        
    """

    # Define the translator
    translator = str.maketrans('', '', string.punctuation + "0123456789")

    # Import the model from hugging face
    # model = SentenceTransformer("all-mpnet-base-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a list of cleaned strings
    list_of_strings = []
    for current_string in data:
        
        #Clean the text
        current_string = current_string.lower().replace("-", " ").translate(translator)     #Lowercase the text, replace "-" with a space, remove punctuation and numbers         
        current_string = " ".join(current_string.split())                                   #Remove all whitespace except for a single space in between words

        # Add the current text to current list
        list_of_strings.append(current_string)

    # Embed the strings
    data_array = model.encode(list_of_strings)
    
    if not similarities:
        return data_array
    else:
        return data_array, model.similarity(data_array, data_array)
