import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


DATA_FILENAME = "../data/random_data3.csv"
DATA_COL_NAME = "data"
CLASSIFICATION_COL_NAME = "classification"      #A new colun in the data frame will be created with this name, and the classifications will be stored here.

CLASSIFICATION_PROMPT = "Give me a summary that is no longer than 5 words of the main subject that the following prompt falls into. Please return nothing else besides the subject."



#Download the data
data = pd.read_csv(DATA_FILENAME, dtype = "object")

#Create a classification column in the data
data[CLASSIFICATION_COL_NAME] = ""

#Get the model and tokenizer location
model_location = "../../data/phi3_model"
tokenizer_location = "../../data/phi3_tokenizer"

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
    "max_new_tokens": 20,
    "return_full_text": False,
    "do_sample": False,
}

#Create the "messages" variable
messages = [
    {"role": "system", "content": "You are a helpful AI summarizer."},
    {"role": "user", "content": ""}
]

#Adjust the classification prompt
CLASSIFICATION_PROMPT = CLASSIFICATION_PROMPT + "\n\nPrompt: " 


#Write a new question for each persona in "data"
for i in tqdm(range(data.shape[0])):
    #Input the prompt into the message
    messages[1]["content"] = CLASSIFICATION_PROMPT + data.loc[i, DATA_COL_NAME]

    #Generate a question from the model
    output = phi_pipeline(messages, **generation_args)[0]["generated_text"]

    #Input the question into the "question" column of the dataset
    data.loc[i,CLASSIFICATION_COL_NAME] = output

    if i % 500 == 0:
        data.to_csv(DATA_FILENAME, index = False)        

#Save the dataset as a csv file
data.to_csv(DATA_FILENAME, index = False)