##I need two main NLP functions, one that can categorise an intent. The next to find specific things in the text, such as the item in question, and the number required.

##To set up the training data, we can write up some patterns and then train a hugging face model on these patterns. 
import json
import pandas as pd
import transformers 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import Dataset

with open('tags.json', 'r')as f:
    intents = json.load(f)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples, padding="max_length", max_length = 5, truncation=True, return_tensors ='pt')


label = []
text = []
#Want to loop through each intent and tokenize the patterns and the tag
for intent in intents['intents']:

    for w in intent['patterns']:
        text.append(w)
        label.append(intent['tag'])


id2label = {0: 'greeting', 1: 'goodbye', 2: 'forecast',3: 'Margin'}
label2id = {'greeting': 0, 'goodbye': 1, 'forecast':2 , 'Margin':3}
'''
for idx , i in enumerate(label):
    label[idx] = label2id[i] 
'''
df=pd.DataFrame()
df['text'] = text
df['labels']=label


#Next step is to call the tokenzier and model
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", max_length = 512)
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels= 4,id2label = id2label, label2id=label2id, problem_type="multi_label_classification")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(df.labels)

X_train, X_test, y_train, y_test = train_test_split(list(df.text), labels, test_size=0.15)


train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
test_encodings =tokenizer(X_test, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

input_ids = train_encodings['input_ids']
attention_masks = train_encodings['attention_mask']
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-6)
from torch.nn import functional as F

test_ids = test_encodings['input_ids']
test_masks = test_encodings['attention_mask']


train_loss = []
test_loss =[]
for epoch in range(3):
    current_loss = 0
    val_loss = 0
    for idx in range(len(input_ids)):
        model.train()
        input_id = input_ids[idx]
        attention_mask = attention_masks[idx]
        label = torch.tensor(y_train[idx]).float()
        label = label.unsqueeze(0)
       # print(label, "!")
        output = model(input_id, attention_mask=attention_mask)
        #print(output.logits, y_train[idx])
        #print(label,label.size(),output.logits.size() ,output.logits.dtype )
        loss = F.cross_entropy( output.logits, label)
        current_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad
    current_loss = current_loss/len(input_ids)
    train_loss.append(current_loss)

    for idx in range(len(test_ids)):
        model.eval()
        input_id = test_ids[idx]
        attention_mask = test_masks[idx]
        output = model(input_id, attention_mask=attention_mask)
        label = torch.tensor(y_test[idx]).float()
        label = label.unsqueeze(0)
        #print(label)
        loss = F.cross_entropy( output.logits, label)
        val_loss += loss.item()
    val_loss = val_loss/len(test_ids)
    test_loss.append(val_loss)


##Find out the accuracy
acc = 0
'''
for idx in range(len(input_ids)):
    input_id = input_ids[idx]
    attention_mask = attention_masks[idx]
    output = model(input_id, attention_mask=attention_mask)
    #print(output.logits.detach())
    probability = torch.softmax(output.logits, dim=1)
    answer= torch.argmax(probability)
    if answer == torch.argmax(torch.tensor(y_train[idx])):
        acc+= 1
'''
for idx in range(len(test_ids)):
    input_id = test_ids[idx]
    attention_mask = test_masks[idx]
    output = model(input_id, attention_mask=attention_mask)
    probability = torch.softmax(output.logits, dim=1)
    answer= torch.argmax(probability)
    if answer == torch.argmax(torch.tensor(y_test[idx])):
        acc+= 1


print(((acc/len(test_ids)))*100 , "%")

#torch.save(model.state_dict(), "./my_new_model")
    
#Plot train val loss
    
import matplotlib.pyplot as plt
plt.plot(train_loss, label = 'Train')
plt.plot(test_loss, label ='Val')
plt.legend()
plt.show()