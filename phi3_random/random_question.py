import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

AMOUNT_OF_DATA = 10
TEMPERATURE = 1
COLUMN_NAME = "data"                            #Name of the column that will store the Phi3 output

SAVE = False                                     #If SAVE = False it will display to the terminal
STORAGE_LOCATION = "../data/random_data.csv"    #Only used if SAVE = True






#Create the Pandas Data Frame
my_series = pd.Series([""]*AMOUNT_OF_DATA, name = COLUMN_NAME, dtype = "object")

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
    "do_sample" : True,
    "temperature" : float(TEMPERATURE)
}

#Create the "messages" variable
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": ""}
]

question = "Output a random question of moderate difficulty from a subject that you are familiar with. Output nothing but the question."

for i in tqdm(range(AMOUNT_OF_DATA)):
    messages[1]["content"] = question
    my_series.at[i] = phi_pipeline(messages, **generation_args)[0]["generated_text"]


if SAVE:
    my_series.to_csv(STORAGE_LOCATION, index = False)
else:
    for i in range(AMOUNT_OF_DATA):
        print(f"{i}: {my_series.at[i]}")
        print()