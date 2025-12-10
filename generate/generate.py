import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from hf_olmo import OLMoTokenizerFast, OLMoForCausalLM
from tqdm import tqdm
from torch.utils.data import Dataset


#Hyperparameters--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AMOUNT_OF_DATA = 2000
TEMPERATURE_PER_ROUND = [.5, 1]                #If you want a single round, set this to a single value inside a list (i.e. [1]). The one I have been doing most of the time is: [2.75, 2, 1.5, 1]
MAX_TOKENS_PER_ROUND = [25, 100]                 #If you want a single round, set this to a single value inside a list (i.e. [200]). The one I have been doing most of the time is: [12, 12, 12, 20]
MAX_LENGTH = 500                                #This is the maximum number of CHARACTERS (not tokens) that the output at each data point will be truncated to.

SAVE = True                                     #If SAVE = True, it saves the data to the STORAGE_LOCATION. If SAVE = False, it prints out the data.
COLUMN_NAME = "text"                            #If SAVE = True, then this is the column name in the dataframe that gets saved
STORAGE_LOCATION = "../../../data/olmo/olmo_2000_generations_temp_.5.csv"    #Only used if SAVE = True

# QUESTION = "<s>"                    #This is the BOS token for the Phi3 model
QUESTION = "<|endoftext|>"          #This it the BOS token for the Olmo model
MESSAGE_TO_PREPEND = " "            #Use this if you are feeding in the BOS token

MODEL = "allenai/OLMo-1B"                     #Either the name of the model or the filepath on your device to where it is saved
TOKENIZER = "allenai/OLMo-1B"                 #Either the name of the tokenizer or the filepath on your device to where it is saved
# MODEL = "microsoft/Phi-3-mini-4k-instruct"
# TOKENIZER = "microsoft/Phi-3-mini-4k-instruct"

# MODEL_ARGS = {"device_map": "auto"}                           #A generic model args
# MODEL_ARGS = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True, "attn_implementation": "flash_attention_2"}     #model args for Phi3 when flash attention is installed
# MODEL_ARGS = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True}                #model args for Phi3 when flash attention is NOT installed
# MODEL_ARGS = {"device_map": "auto", "torch_dtype": torch.bfloat16, "attn_implementation" : "flash_attention_2"}                       #model args for Gemma
# MODEL_ARGS = {"device_map": "auto", "torch_dtype": torch.bfloat16}                                                                    #model args for Llama
# MODEL_ARGS = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True}                                                 #model args for openelm
MODEL_ARGS = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True}                                                 #model args for OLMo

MESSAGES = None                      #1 for system then user prompt (for chatbot trained models like Phi3), 0 for just user (for models not trained in chat format), put None for the message to be a simple string and not a list (for tiny stories model or olmo)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Create the Pandas Data Frame
if COLUMN_NAME is None:
    COLUMN_NAME = "text"
my_series = pd.Series([""]*AMOUNT_OF_DATA, dtype = "object", name = COLUMN_NAME)

#Download the model and tokenizer
if MODEL[:7].lower() == 'allenai':
    model = OLMoForCausalLM.from_pretrained(MODEL, **MODEL_ARGS)
    tokenizer = OLMoTokenizerFast.from_pretrained(TOKENIZER)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, **MODEL_ARGS)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

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
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": ""}
    ]
elif MESSAGES == 0:
    messages = [{"role": "user", "content": ""}]

#Do a seperate generation loop for each different temperature
for i in range(len(TEMPERATURE_PER_ROUND)):
    generation_args = {
        "max_new_tokens": MAX_TOKENS_PER_ROUND[i],
        "return_full_text": False,
        "do_sample" : True,
        "temperature" : float(TEMPERATURE_PER_ROUND[i])
    }
    print()
    print(f"Generating {generation_args["max_new_tokens"]} new tokens at temperature {generation_args["temperature"]}")

    #For the given temperature, loop through all of your data and generate the text
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


#Truncate the series to include only up to the first question mark, or up to the MAX_LENGTH number of characters, whichever is lower
my_series = my_series.str[:MAX_LENGTH]         
            
#Save or output the data
if SAVE:
    my_series.to_frame().to_csv(STORAGE_LOCATION, index = False)
else:
    for i in range(AMOUNT_OF_DATA):
        print(f"{i}: {my_series.at[i]}")
        print()