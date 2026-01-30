# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 23:04:31 2026

@author: Lehlohonolo Saohatse
"""

'''
1. Apply One-Hot Encoding and Label Encoding to a dataset with categorical c=variables
2. Experiment with different encoding techniques and observe their impact on model performance
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/Lehlohonolo-Saohatse/master-ml-dl-ai-agents-tensorflow-pytorch-projects/refs/heads/main/data/titanic.csv"
df = pd.read_csv(url)

# Display Dataset Information
print("Dataset Information:")
print(df.info())

# Preview the First few rows
print("\n Dataset Priview:")
print(df.head())

# Apply One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first = True)

# Display Encoded Dataset
print("\n One-Hot Encoded Dataset:")
print(df_one_hot.head())

# Apply Label Encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display Encoded Dataset
print("\n Label Encoded Dataset:")
print(df[['Pclass', 'Pclass_encoded']].head())

# Apply Frequency Encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display Frequency Encoded Feature
print("\n Frequency ENcoded Feature:")
print(df[['Ticket', 'Ticket_frequency']].head())

X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Age'])
y = df['Survived']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy With One-Hot Encoding: ", accuracy_score(y_test, y_pred))


