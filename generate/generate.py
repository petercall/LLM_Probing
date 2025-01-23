import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

AMOUNT_OF_DATA = 10000
NUM_ROUNDS = 4                                  #If you just want a single round, set this to 1
TEMPERATURE_PER_ROUND = [2.75, 2, 1.5, 1]       #If NUM_ROUNDS = 1, set this to a single value inside a list (i.e. [1])
MAX_TOKENS_PER_ROUND = [12, 12, 12, 20]            #If NUM_ROUNDS = 1, set this to a single value inside a list (i.e. [200])
MAX_LENGTH = 180

BATCH = False                                  #Whether to do batch processing or not. I haven't completely tested this yet, so put BATCH = False for now.
BATCH_SIZE = 8                                 #Ignored if BATCH = False

SAVE = True                                     #If SAVE = True, it saves the data to the STORAGE_LOCATION. If SAVE = False, it prints out the data.
STORAGE_LOCATION = "../data/variableTemp_tinyStories.csv"    #Only used if SAVE = True

QUESTION = "Output a question that you know something about. Output nothing but the question."
MESSAGE_TO_PREPEND: str = "Finish the following question by picking up right where it left off: "   #Make sure to end this with a space

MODEL_LOCATION = None                            #The location of the model OR NONE (Phi3 is at: "../../data/phi3_model")
MODEL = "roneneldan/TinyStories-33M"        #The model to use if MODEL_LOCATION is None. Ignored if MODEL_LOCATION is not None

TOKENIZER_LOCATION = None                        #The location of the tokenizer OR NONE (Phi3 is at: "../../data/phi3_tokenizer")  
TOKENIZER = "roneneldan/TinyStories-33M"    #The tokenizer to use if TOKENIZER_LOCATION is None. Ignored if TOKENIZER_LOCATION is not None

# MODEL_ARGS = {"device_map": "auto"}
MODEL_ARGS = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True, "attn_implementation": "flash_attention_2"}     #model args for Phi3
# MODEL_ARGS = {"device_map": "auto", "torch_dtype": torch.bfloat16, "attn_implementation" : "flash_attention_2"}                       #model args for Gemma
# MODEL_ARGS = {"device_map": "auto", "torch_dtype": torch.bfloat16}                                                                    #model args for Llama
# MODEL_ARGS = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True}                                                 #model args for openelm

MESSAGES = None                      #1 for system then user prompt, 0 for just user, put None for the message to be a simple string and not a list





#Create the Pandas Data Frame
if not BATCH:
    my_series = pd.Series([""]*AMOUNT_OF_DATA, dtype = "object", name = "answer")
else:
    ser1 = pd.Series([""]*AMOUNT_OF_DATA)
    ser2 = pd.Series([QUESTION]*AMOUNT_OF_DATA)
    my_df = pd.DataFrame({"answer": ser1, "question":ser2})

#Get the model and tokenizer location
if MODEL_LOCATION is None:
    model_location = MODEL
else:
    model_location = MODEL_LOCATION

if TOKENIZER_LOCATION is None:
    tokenizer_location = TOKENIZER
else:
    tokenizer_location = TOKENIZER_LOCATION

#Download the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_location, **MODEL_ARGS)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)

#For llama, you have to set the pad token id to be the end of sequence token id
if "llama" in MODEL.lower() or "stories" in MODEL.lower():
    tokenizer.pad_token_id = tokenizer.eos_token_id

#Input the model and tokenizer into a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

#Create the "messages" variable
if MESSAGES == 1:
    messages = [
        {"role": "system", "content": "You are an AI question generator."},
        {"role": "user", "content": ""}
    ]
elif MESSAGES == 0:
    messages = [{"role": "user", "content": ""}]

#Create the dataset for your class
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

if BATCH:
    my_dataset = MyDataset(my_df["question"])


for i in range(NUM_ROUNDS):
    generation_args = {
        "max_new_tokens": MAX_TOKENS_PER_ROUND[i],
        "return_full_text": False,
        "do_sample" : True,
        "temperature" : float(TEMPERATURE_PER_ROUND[i])
    }

    if not BATCH:
        for j in tqdm(range(AMOUNT_OF_DATA)):
            if i == 0:
                if MESSAGES is not None:
                    messages[MESSAGES]["content"] = QUESTION
                else:
                    messages = QUESTION
                
                my_series.at[j] = pipe(messages, **generation_args)[0]["generated_text"]
            else:
                if MESSAGES is not None:
                    messages[MESSAGES]["content"] = MESSAGE_TO_PREPEND + my_series.at[j]
                else:
                    messages = MESSAGE_TO_PREPEND + my_series.at[j]
                
                my_series.at[j] = my_series.at[j] + " " + pipe(messages, **generation_args)[0]["generated_text"]
    else:
        for i, output in enumerate(tqdm(pipe(my_dataset, batch_size = BATCH_SIZE, **generation_args), total = len(my_dataset))):
            if i == 0:
                my_df.at[i, 'answer'] = output[0]["generated_text"]
                my_df["question"] = MESSAGE_TO_PREPEND + my_df.at[i, "answer"]
                print(my_df.at[i, "answer"])
                print(my_df.at[i, "question"])
                print()
            else:
                my_df.at[i, "answer"] = my_df["answer"][i] + " " + output[0]["generated_text"]
                my_df.at[i, "question"] = MESSAGE_TO_PREPEND + my_df["answer"][i]
                print(my_df.at[i, "answer"])
                print(my_df.at[i, "question"])
                print()

if BATCH:
    my_series = my_df["answer"]

#Truncate the series to include only up to the first question mark, or up to the MAX_LENGTH number of characters, whichever is lower
my_series = (my_series.str.split("?", expand = True)[0] + "?").str[:MAX_LENGTH]         
            
#Save or output the data
if SAVE:
    my_series.to_csv(STORAGE_LOCATION, index = False)
else:
    for i in range(AMOUNT_OF_DATA):
        print(f"{i}: {my_series.at[i]}")
        print()