'''
These packages must be installed from the Python console:

import pip
pip.main(['install','ucimlrepo'])
pip.main(['install', 'scikit-learn'])
pip.main(['install','pandas'])
pip.main(['install', 'seaborn'])
pip.main(['install', 'matplotlib'])
pip.main(['install', 'certifi'])

'''
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mushroom = fetch_ucirepo(name='Mushroom')

X = mushroom.data.features
y = mushroom.data.targets
# 70/30 split for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Encoding required for categorical features
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_encoded, y_train)

def predict_edibility():
    inputs = []
    for var in entry_vars:
        input_value = var.get()
        inputs.append(input_value)

    inputs_array = np.array([inputs])
    # encoding is again required since the 22-tuple input represents
    # the categorical values for the 22 feature variables
    inputs_encoded = encoder.transform(inputs_array)

    prediction = nb_model.predict(inputs_encoded)[0]
    if prediction == 0:
        messagebox.showinfo("Prediction", "The mushroom is predicted to be edible.")
    else:
        messagebox.showinfo("Prediction", "The mushroom is predicted to be poisonous.")

root = tk.Tk()
root.title("Mushroom Edibility Predictor")

entry_vars = []
for i in range(22):
    label = tk.Label(root, text=f"Feature {i + 1}:")
    label.grid(row=i, column=0, sticky="w")
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var)
    entry.grid(row=i, column=1)
    entry_vars.append(entry_var)

predict_button = tk.Button(root, text="Predict Edibility", command=predict_edibility)
predict_button.grid(row=22, columnspan=2)

root.mainloop()
