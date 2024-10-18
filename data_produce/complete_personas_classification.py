import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

#Download the data
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
    "max_new_tokens": 20,
    "return_full_text": False,
    "do_sample": False,
}

#Create the "messages" variable
messages = [
    {"role": "system", "content": "You are a helpful AI summarizer."},
    {"role": "user", "content": ""}
]

#Add in the classification to the "answer" category
classification_prompt = "Give me a summary that is no longer than 5 words of the main subject that the following prompt falls into. Please return nothing else besides the subject.\n\nPrompt: "

#Get the start and end values for the questions to loop through
start = complete_personas["classification"].count()
end = complete_personas.shape[0]

#Write a new question for each persona in "complete_personas"
for i in tqdm(range(start, end)):
    #Input the prompt into the message
    messages[1]["content"] = classification_prompt + complete_personas.loc[i, "answer"]

    #Generate a question from the model
    output = phi_pipeline(messages, **generation_args)[0]["generated_text"]

    #Input the question into the "question" column of the dataset
    complete_personas.loc[i,"classification"] = output

    if i % 450 == 0:
        complete_personas.to_csv("complete_personas.csv", index = False)        

#Save the dataset as a csv file
complete_personas.to_csv("complete_personas.csv", index = False)