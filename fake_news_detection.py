import pandas as pd                                                  #import libraries,last=give fucntion like softmax
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split                 #split dataset in to train and test
from sklearn.metrics import accuracy_score, mean_squared_error       #calculating accuracy and mse

from transformers import (
    BertTokenizer,                                                   #convert text in to number(token)  
    BertForSequenceClassification,                                   #classification real/fake
    Trainer,                                                         #handle training automatically
    TrainingArguments                                                #settings for training
)

from datasets import Dataset                                         #convert pd dataframe into hugging face dataset format

fake_df = pd.read_csv(                                               #load dataset
    "Fake.csv",
    engine="python",
    on_bad_lines="skip"
)

true_df = pd.read_csv(                                              #load datset
    "True.csv",
    engine="python",
    on_bad_lines="skip"
)

fake_df["label"] = 0                                               #labeling data
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0)                         #combining dataset

df = df.sample(frac=1, random_state=42)                            #shuffle datset

df = df.sample(3000)                                               #take only 3000  rows reduce training time

df["content"] = df["title"] + " " + df["text"]                     # for input text (title+articletext)

X = df["content"]                                                  #splitting feature(x=input,Y=output(fake/real))
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(               #train,test,split
    X,
    y,
    test_size=0.2,                                                 #80 perc training and 20 perc testing
    random_state=42
)

train_df = pd.DataFrame({                                          #convert in to format of hugging face dataset
    "text": x_train,
    "label": y_train
})

test_df = pd.DataFrame({                                                   
    "text": x_test,
    "label": y_test
})

train_dataset = Dataset.from_pandas(train_df)                      #format for transformers library
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")    #load bert tokenizer (tiny bert model)

def tokenize_function(example):                                     #tokenization func so that (numbers)bert can understand
         
    return tokenizer(
        example["text"],
        padding="max_length",                                        #make all inputs same length
        truncation=True,                                             #cut long text
        max_length=64                                               #64 tokens used
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)    #apply tokenization on whole datset
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]                  #token numbers,which words matter,fake/real
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

model = BertForSequenceClassification.from_pretrained(                #loads BERT model,add classification layer and ,output=2classes

    "bert-base-uncased",
    num_labels=2
)

def compute_metrics(eval_pred):                                     #evaluation function

    logits, labels = eval_pred                                      #logits=raw model output,argmax=pick highest probability clas

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)                  #correct predictions 

    mse = mean_squared_error(labels, predictions)                   #error value

    return {
        "accuracy": accuracy,
        "mse": mse
    }

training_args = TrainingArguments(                                 #training setting
    output_dir="./results",             
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    logging_steps=20
)

trainer = Trainer(                                                #training setup
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()                                                   #train model

results = trainer.evaluate()                                      #evaluate model on unseen data

print("\nMODEL EVALUATION")

print(f"Accuracy : {results['eval_accuracy']:.4f}")

print(f"MSE : {results['eval_mse']:.4f}")

def predict_news(news_text):                                    #prediction func

    inputs = tokenizer(                                        #converts input into BERT format tensors
        news_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    outputs = model(**inputs)                                   #runs model prediction

    prediction = torch.argmax(outputs.logits, dim=1).item()     #decides fake real

    probabilities = F.softmax(outputs.logits, dim=1)             #converts output into probability

    confidence = torch.max(probabilities).item() * 100          #gives confidence score

    print("\nNEWS ARTICLE:")                                    #print result
    print(news_text)

    print("\nRESULT")

    if prediction == 0:

        print("FAKE NEWS")

    else:

        print("REAL NEWS")

    print(f"\nConfidence Score: {confidence:.2f}%")

news = input("\nEnter News Text:\n\n")                         #user input

predict_news(news)                                             #model predicts