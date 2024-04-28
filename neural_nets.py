# -*- coding: utf-8 -*-
"""Neural Nets.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jEr5KfQ7c4t7rmwi8s_gKYaiRmEj6vIB
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

Data= pd.read_csv("/content/Wholedata.csv")

Data.head()

Data.dropna(inplace=True)
Data = Data.drop(["Date"], axis=1)

Data

encoded_df = pd.DataFrame(y, columns=['Control Campaign', 'Test Campaign'])

# Concatenate the original DataFrame (without the label column) with the new DataFrame of encoded labels
result_df = pd.concat([Data.iloc[:, 1:], encoded_df], axis=1)

result_df

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Assume 'data' is your DataFrame name
X = Data.iloc[:, 1:]  # Exclude the 'Campaign Name' column for features
y = Data['Campaign Name']  # Use 'Campaign Name' as the label

# Convert labels to numerical format
label_mapping = {'Control Campaign': 0, 'Test Campaign': 1}
y = y.map(label_mapping)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pd.concat([X_train, pd.DataFrame(y_train, columns=["Control Campaign",'Test Campaign'])])

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(32, activation='relu'),  # Additional hidden layer
    Dense(2, activation='softmax')  # Output layer for 2 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

import numpy as np

# Predict classes on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()