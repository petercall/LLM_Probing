import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


#Select column of interest
COL = "answer"
NEW_COL_NAME = "phi3_classification_cluster"





#Download the data and change the question, answer, and classification columns to be object.
complete_personas = pd.read_csv("../data/complete_personas.csv", dtype = {"persona" : "object", "question" : "object", "answer" : "object", "classification" : "object", "label" : "int", "label2" : "int", "cluster_30000" : "int", "cluster_30000_2qr" : "int", "cluster_30000_2kmeans" : "int"})
X = complete_personas[COL]
complete_personas[NEW_COL_NAME] = " "

#Get the model and tokenizer location
model_location = "../../phi3_model"
tokenizer_location = "../../phi3_tokenizer"

#Download the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_location,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)

#Input the model and tokenizer into a pipeline
phi_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

#Create the arguments you want to be input into the model when calling it
generation_args = {
    "max_new_tokens": 200,
    "return_full_text": False,
    "do_sample": False,
}

#Create the "messages" variable
message = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": ""}
]

#Create the prompt list
# prompt = "Please classify the following prompt into one of the following categories: [mathematics, coding, reasoning, science, history, arts, daily activities]. Please return no additional text but the category the prompt falls into."
prompt = "Please classify the following prompt into one of the following categories: [sustainability, work, literature, education, sports, history, art, data science]. Please return no additional text but the category the prompt falls into."

for i in tqdm(range(X.shape[0])):
    #Input the question into the message variable
    message[1]["content"] = prompt + " Prompt: " + X[i]
    
    #Get Phi 3's output
    output = phi_pipeline(message, **generation_args)[0]["generated_text"]

    #Save Phi 3's output in the new column
    complete_personas.at[i, NEW_COL_NAME] = output      

#Save the dataset as a csv file
complete_personas.to_csv("../data/complete_personas_cluster.csv", index = False)