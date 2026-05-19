import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

real_path = "real"
fake_path = "fake"

images = []
labels = []

img_size = 128

def load_images(folder, label):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is not None:

            img = cv2.resize(img, (img_size, img_size))

            images.append(img)

            labels.append(label)

load_images(real_path, 0)
load_images(fake_path, 1)

X = np.array(images) / 255.0
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(128,128,3)
))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(
    128,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

predictions = model.predict(x_test)

predictions = (predictions > 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

image_path = input("\nEnter Image Path:\n")

img = cv2.imread(image_path)

img_resized = cv2.resize(img, (128,128))

img_array = np.array(img_resized) / 255.0

img_array = np.expand_dims(img_array, axis=0)

result = model.predict(img_array)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.axis("off")

if result[0][0] > 0.5:

    plt.title("FAKE IMAGE")

else:

    plt.title("REAL IMAGE")

plt.show()