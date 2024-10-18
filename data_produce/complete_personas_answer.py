import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data import Dataset

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

#Define the custom dataset
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        return self.data[i]
    
start = complete_personas["answer"].count()
total = complete_personas.shape[0] - start
dataset = MyDataset(complete_personas["question"][start:].reset_index(drop = True))

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

#Cycle through each pre-written question and generate an answer from the Phi 3 Mini model. Save the answer in the "answer" column of the question_group data frame
for i, out in enumerate(tqdm(phi_pipeline(dataset, batch_size = 30, **generation_args), total = total)):
    complete_personas.loc[start + i, "answer"] = out[0]["generated_text"]
    
    if i % 330 == 0:
        complete_personas.to_csv("complete_personas.csv", index = False)  
    
# Save the dataset as a csv file
complete_personas.to_csv("complete_personas.csv", index = False)