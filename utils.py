##This file will implement functions that use the intent classifer model. The first needs to be a prediction method
import torch 
import transformers 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from LatestPriceDatabaseBuilder import Data
import json
import random
from forecaster import forecast
import string as st
with open('tags.json', 'r')as f:
    intents = json.load(f)

def responder(string):
    '''A function that coordinates the correct response'''

    string = string.translate(str.maketrans('', '', st.punctuation))

    intent = classifier(string)

    if intent == "greeting":
            return random.choice(intents['intents'][0]['responses'])
    elif intent == "goodbye":
        return random.choice(intents['intents'][1]['responses'])
    elif intent == "forecast":
        item_name = identify_item(string)
       
        if item_name == "Could find no item":
            return "Could find no item"
    
        n_of_hrs = identify_hours(string)
      
        if n_of_hrs == "I couldn't identify how many hours you would like the item forecasted for. Can you try rephrasing so that a number is directly before hour/hours?":
            return "I couldn't identify how many hours you would like the item forecasted for. Can you try rephrasing so that a number is directly before hour/hours?"
       
        avghighprice, avglowprice = forecast(item_name,n_of_hrs)

        return (f"After {n_of_hrs} hrs, {item_name} will have an average high price of {avghighprice} and an average low price of {avglowprice}")
    
    elif intent == "margin":

        threshold = identify_threshold(string)
        if threshold == "I couldn't find a threshold, make sure you include the keyword gp. I do account for values like 17k, but they must be directly before the gp i.e 17k gp":
            return "I couldn't find a threshold, make sure you include the keyword gp. I do account for values like 17k, but they must be directly before the gp i.e 17k gp"
        
        TopN = identify_topN(string)
        return MarginSorter(threshold, TopN)
    
def MarginSorter(threshold, TopN):
    df = Data.Sheet
    df = df.sort_values(by = 'Margin', ascending = False)

    return (df.loc[df['BuyPrice'] <= threshold]).head(TopN)

def classifier( string):

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels= 4, problem_type="multi_label_classification")
    model.load_state_dict(torch.load(".\my_new_model"))
    model.eval()



    #Need to tokenize the string
    encoding = tokenizer(string,truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    #Next take the input_ids and attention mask and compute the output
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)
    #Finally determine the classification
    probability = torch.softmax(output.logits.detach(), dim=1)
    answer= torch.argmax(probability)
    if answer == torch.argmax(torch.tensor([1,0,0,0])):
        return "margin"
    elif answer == torch.argmax(torch.tensor([0,1,0,0])):
        return "forecast"
    elif answer == torch.argmax(torch.tensor([0,0,1,0])):
        return "goodbye"
    elif answer == torch.argmax(torch.tensor([0,0,0,1])):
        return "greeting"

def identify_item(string):
    # ""What is your prediction on the price of a dragon dagger in the next hour""
    df = Data.Sheet
    for s in string.split(" "):
        #print(s)
        for x in df.ItemName:
            if x == s:
                #print("Found ya ")
                return s
    word = string.split(" ")
    sieve = []
    for idx , x in enumerate(word):
        if idx+1 == len(word):
            for s in sieve:
                for x in df.ItemName:
                    m = x.replace('"','' ).lower()
                    if m == s.lower():
                        #print("Found ya ")
                        return x
            break
        sieve.append(word[idx]+" " + word[idx+1])
    
    return "Could find no item"

def identify_hours(string):
    #"What do you forecast the price of a bronze pickaxe to be in 3 hours."
    #"What is your prediction on the price of a dragon dagger in the next hour"
    string = string.lower()
    #All we need to do is iterate through the string to find a word that describes time. Lets look for the key words hour or day.
    for idx, s in enumerate(string.split(" ")): 
        if s== "hour":
           number = int(string.split(" ")[idx-1].strip())
           return number
        elif s == "hours":
            number = int(string.split(" ")[idx-1].strip())
            return number
    
    return "I couldn't identify how many hours you would like the item forecasted for. Can you try rephrasing so that a number is directly before hour/hours?"

def identify_threshold(string):
    string = string.lower()
    for idx, s in enumerate(string.split(" ")): 
        if s== "gp":
           #print(string.split(" ")[idx-1].split("k"))
           if len(string.split(" ")[idx-1].split("k")) > 1:
               return (int(string.split(" ")[idx-1].split("k")[0])*1000)
           else:
               return (int(string.split(" ")[idx-1].split("k")[0]))

    return "I couldn't find a threshold, make sure you include the keyword gp. I do account for values like 17k, but they must be directly before the gp i.e 17k gp"

def identify_topN(string):
    #This function identifies which how many entries wish to be seen in the margin sorter. I.e top 5 top 15 etc.
    string = string.lower()
    for idx, s in enumerate(string.split(" ")): 
        if s== "top":
           return int(string.split(" ")[idx+1])
    print("As the number for amount of entries hasnt been specified, I will default to 5.", 
          "If you want a different amount of entries make sure to include the keyword top and then an integer after it. I.e show me top 10 ...")
    return 5


 

