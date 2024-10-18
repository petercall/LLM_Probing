import numpy as np
import pandas as pd


#Get the personas dataset and rename the column to "input persona"
persona_df = pd.read_json("hf://datasets/proj-persona/PersonaHub/persona.jsonl", lines=True)

#Grab the rest of the dataset, which I will call the "question group"
math_df = pd.read_json("hf://datasets/proj-persona/PersonaHub/math.jsonl", lines=True)
reasoning_df = pd.read_json("hf://datasets/proj-persona/PersonaHub/reasoning.jsonl", lines=True)
instruction_df = pd.read_json("hf://datasets/proj-persona/PersonaHub/instruction.jsonl", lines=True)
knowledge_df = pd.read_json("hf://datasets/proj-persona/PersonaHub/knowledge.jsonl", lines=True)
npc_df = pd.read_json("hf://datasets/proj-persona/PersonaHub/npc.jsonl", lines=True)

#Concatenate all of the question group data frames together
question_group = pd.concat([math_df, reasoning_df, instruction_df, knowledge_df, npc_df], axis = 0, ignore_index = True)

#Rename the "input persona" column to just "persona"
question_group.rename({"input persona" : "persona"}, axis = 1, inplace = True)

#Drop the "description" column and rename the "synthesized text" column to "question"
question_group.drop("description", axis = 1, inplace = True)
question_group.rename({"synthesized text" : "question"}, axis = 1, inplace = True)

#Create empty "answer" and "classification" columns
question_group["answer"] = None
question_group["classification"] = None

#Get the unique personas from the question group
unique_question_personas = question_group["persona"].drop_duplicates(ignore_index = True).to_frame()

#Join in any of the unique personas from persona_df
complete_personas = pd.concat([unique_question_personas, persona_df], ignore_index = True, axis = 0).drop_duplicates(ignore_index = True)

#Create empty "question", "answer", and "classification" columns
complete_personas["question"] = None
complete_personas["answer"] = None
complete_personas["classification"] = None


####


#Create the "starts_with" Series for question_group[persona]
starts_with = question_group["persona"].str[0].str.lower().value_counts()
my_list = [0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,26,28,34,36,38,44,47,51,75,93,102,132,147,166]
starts_with = starts_with.iloc[my_list].index.to_list()

#Delete any entries that don't start with first leter in "starts_with"
bool_series = pd.Series(np.zeros(question_group.shape[0])).astype(bool)
first_letter = question_group["persona"].str[0].str.lower()
for i in range(question_group.shape[0]):
    if first_letter[i] in starts_with:
        bool_series.iloc[i] = True

question_group = question_group.loc[bool_series, :].reset_index(drop = True) 


#Create the "starts_with" Series for question_group[question]
starts_with2 = question_group["question"].str[0].str.lower().value_counts()
my_list2 = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,25,30,32,39,41,43,51,56,63,68,70,73,79,85,88,96,98,101,107,125,132]
starts_with2 = starts_with2.iloc[my_list2].index.to_list()

#Delete any entries that don't start with first leter in "starts_with"
bool_series2 = pd.Series(np.zeros(question_group.shape[0])).astype(bool)
first_letter2 = question_group["question"].str[0].str.lower()
for i in range(question_group.shape[0]):
    if first_letter2[i] in starts_with2:
        bool_series2.iloc[i] = True

question_group = question_group.loc[bool_series2, :].reset_index(drop = True)


#Create the "starts_with" Series for complete_personas[persona]
starts_with3 = complete_personas["persona"].str[0].str.lower().value_counts()
my_list3 = [0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,26,31,35,41,44,48,54,56,60,61,66,78,95,98,99,139,152,155,158,163]
starts_with3 = starts_with3.iloc[my_list3].index.to_list()

#Delete any entries that don't start with first leter in "starts_with"
bool_series3 = pd.Series(np.zeros(complete_personas.shape[0])).astype(bool)
first_letter3 = complete_personas["persona"].str[0].str.lower()
for i in range(complete_personas.shape[0]):
    if first_letter3[i] in starts_with3:
        bool_series3.iloc[i] = True

complete_personas = complete_personas.loc[bool_series3, :].reset_index(drop = True)


#Drop the values where the persona is "N/A" (index 87336 for complete_personas and index 88226 for question group)
complete_personas = complete_personas.loc[ ~(complete_personas["persona"].str[:] == "N/A") , :]
question_group = question_group.loc[ ~(question_group["persona"].str[:] == "N/A") , :]


# Export the two data frames
question_group.to_csv("question_group.csv", index = False)
complete_personas.to_csv("complete_personas.csv", index = False)

print("Done!")