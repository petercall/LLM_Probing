import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

#Download the data and change the question, answer, and classification columns to be object.
complete_personas = pd.read_csv("complete_personas.csv", dtype = "object")

#Get the model and tokenizer location
model_location = "../phi3_model"
tokenizer_location = "../phi3_tokenizer"

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
messages_prewritten = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": ""}
]


#Create the prompt list
prompt_list = ["Write a question that someone with the following persona would ask:\n\t", ""]

#Create the options to add to the prompt
options = [
    "Make sure the following things are true:\n",
    "\tThe text the model returns is only a single question with no text before or after.\n",
    "\tMake the question reasonably complicated.\n"
]

#Create the "messages" variable
messages_generated = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": ""}
]


#Generate a new question for each persona in complete_persona

#Get the start and end values for the questions to loop through
start = complete_personas["question"].nunique()
# end = 10
end = complete_personas.shape[0]

#Write a new question for each persona in "complete_personas"
for i in tqdm(range(start, end)):
    #Get the persona and input it into the prompt
    persona = complete_personas["persona"][i]
    prompt_list[1] = persona + "\n\n"

    #Create the prompt
    if len(options) > 1:
        prompt = "".join(prompt_list) + "".join(options)
    else:
        prompt = "".join(prompt_list)

    #Input the prompt into the message
    messages_generated[1]["content"] = prompt

    #Generate a question from the model
    output = phi_pipeline(messages_generated, **generation_args)[0]["generated_text"]

    #Input the question into the "question" column of the dataset
    complete_personas.loc[i,"question"] = output

    if i % 50 == 0:
        complete_personas.to_csv("complete_personas.csv", index = False)        

#Save the dataset as a csv file
complete_personas.to_csv("complete_personas.csv", index = False)