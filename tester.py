import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import pipeline



#Download the data
data = pd.read_csv("data/random_data3.csv", dtype = "object")

new_data = data["data"].sample(10)
for val in new_data:
    print(val)
    print()


# print(data.info())
# print(data["data_supervised_labels"].value_counts()*100 / data.shape[0])

