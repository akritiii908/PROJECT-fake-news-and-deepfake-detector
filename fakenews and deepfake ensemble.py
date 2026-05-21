

# IMPORT LIBRARIES


import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# LOAD FAKE NEWS DATASET


fake_df = pd.read_csv(
    "Fake.csv",
    engine="python",
    on_bad_lines="skip"
)

true_df = pd.read_csv(
    "True.csv",
    engine="python",
    on_bad_lines="skip"
)

# LABELS
fake_df["label"] = 0
true_df["label"] = 1

# COMBINE DATA
df = pd.concat([fake_df, true_df], axis=0)

# SHUFFLE
df = df.sample(frac=1, random_state=42)

# SMALL SAMPLE FOR FAST TRAINING
df = df.sample(3000)

# CREATE INPUT TEXT
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CONVERT TO DATAFRAME
train_df = pd.DataFrame({
    "text": x_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text": x_test,
    "label": y_test
})

# CONVERT TO HUGGINGFACE DATASET
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# LOAD TOKENIZER


tokenizer = BertTokenizer.from_pretrained(
    "prajjwal1/bert-tiny"
)


# TOKENIZATION


def tokenize_function(example):

    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# FORMAT FOR PYTORCH
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


# LOAD BERT MODEL


bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# METRICS


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy
    }


# TRAINING SETTINGS


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_steps=20
)


# TRAINER


trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# TRAIN BERT


trainer.train()


# CNN DATASET


real_path = "real"
fake_path = "fake"

images = []
labels = []

img_size = 128


# LOAD IMAGES


def load_images(folder, label):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is not None:

            img = cv2.resize(
                img,
                (img_size, img_size)
            )

            images.append(img)

            labels.append(label)

# REAL = 0
load_images(real_path, 0)

# FAKE = 1
load_images(fake_path, 1)

# NORMALIZE
X = np.array(images) / 255.0
y = np.array(labels)

# SPLIT
x_train_img, x_test_img, y_train_img, y_test_img = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# CNN MODEL


cnn_model = Sequential()

cnn_model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(128,128,3)
))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(
    128,
    (3,3),
    activation='relu'
))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Flatten())

cnn_model.add(Dense(
    128,
    activation='relu'
))

cnn_model.add(Dropout(0.5))


cnn_model.add(Dense(
    1,
    activation='sigmoid'
))


# COMPILE MODEL


cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# TRAIN CNN


cnn_model.fit(
    x_train_img,
    y_train_img,
    epochs=5,
    batch_size=32,
    validation_data=(x_test_img, y_test_img)
)


# BERT PREDICTION FUNCTION
def predict_news_score(news_text):

    inputs = tokenizer(
        news_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    outputs = bert_model(**inputs)

    probabilities = F.softmax(
        outputs.logits,
        dim=1
    )

    fake_score = probabilities[0][0].item()

    real_score = probabilities[0][1].item()

    return fake_score, real_score

# CNN PREDICTION FUNCTION


def predict_image_score(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img, (128,128))

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    result = cnn_model.predict(img)

    fake_score = result[0][0]

    real_score = 1 - fake_score

    return fake_score, real_score, img


# ENSEMBLE PREDICTION


def ensemble_prediction(news_text, image_path):

    # TEXT SCORES
    text_fake, text_real = predict_news_score(
        news_text
    )

    # IMAGE SCORES
    image_fake, image_real, original_img = predict_image_score(
        image_path
    )

    # FUSION
    final_fake_score = (
        text_fake + image_fake
    ) / 2

    final_real_score = (
        text_real + image_real
    ) / 2

    # PRINT RESULTS
    print("\n========== RESULTS ==========")

    print("\nTEXT FAKE SCORE :", text_fake)

    print("IMAGE FAKE SCORE:", image_fake)

    print("\nFINAL FAKE SCORE :", final_fake_score)

    print("FINAL REAL SCORE :", final_real_score)

    # FINAL DECISION
    if final_fake_score > final_real_score:

        print("\nFINAL RESULT : FAKE")

    else:

        print("\nFINAL RESULT : REAL")

    # SHOW IMAGE
    img_display = cv2.imread(image_path)

    plt.imshow(
        cv2.cvtColor(
            img_display,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")

    if final_fake_score > final_real_score:

        plt.title("FAKE")

    else:

        plt.title("REAL")

    plt.show()


# USER INPUT

news = input("\nEnter News Text:\n\n")

image_path = input("\nEnter Image Path:\n\n")

# FINAL PREDICTION


ensemble_prediction(
    news,
    image_path
)