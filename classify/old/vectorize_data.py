from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer



def vectorize_data(data, ngram = 1, min_df = 1, max_df = .7, num_sing_vectors = 0):
    """
    Turn a Pandas Series of text into an array of data by performing tf_idf on the text.
    If num_sing_vectors != 0, then it also uses PCA and projects on the top num_sing_vectors number of singular vectors.
    
    Inputs:
        data: Pandas Series (length = n, dtype = object)
            This is the column of data that you want to turn into a Numpy array.
        
        ngram: int (default = 1)
            If ngram = 1 then the vectorizer will only include single words. If ngram = 2 it will include bigrams as well.
        
        min_d: int (default = 1)
            This is the number of documents that a term must appear in in order to be included by the vectorizer.
        
        max_df: float (default = .7)
            If a word appears in this number of documents or more, it is counted as a top word and is not included in the vectorizer.
            
        num_sing_vectors: int (defualt = 0)
            This is the number of singular vectors that we will use in the PCA transformation of the data.
            If num_sing_vectors = 0, it will not perform PCA.
    
    Returns:
        output: Scipy Sparse Matrix
            The output has length n and is a matrix where each row has been converted to a vector.
    
    """
    
    #Define the TFID vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range = (1,ngram),
        min_df = min_df,
        max_df = max_df
    )

    #Apply tfidf to the database
    X = vectorizer.fit_transform(data)

    #Define and apply the LSA
    if num_sing_vectors != 0:
        lsa = make_pipeline(TruncatedSVD(n_components=num_sing_vectors), Normalizer(copy=False))
        X = lsa.fit_transform(X)
    
    return X