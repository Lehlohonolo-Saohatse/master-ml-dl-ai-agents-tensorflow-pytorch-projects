# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:16:53 2026

@author: Lehlohonolo Saohatse
"""

import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/Lehlohonolo-Saohatse/master-ml-dl-ai-agents-tensorflow-pytorch-projects/refs/heads/main/data/titanic.csv"
df = pd.read_csv(url)

# Display dataset Information
print("Dataset Info: \n")
print(df.info())

# Preview the first few rows
print("\n Dataset Preview: \n")
print(df.head())

# Separate Features
categorical_features = df.select_dtypes(include=["object"]).columns
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

print("\n categorical Features: ", categorical_features.tolist())
print("\n Numerical Features: ", numerical_features.tolist())

# Display Summary of Categorical Features
print("\n Categorical Feature Summary: \n")
for col in categorical_features:
    print(f"{col}: \n", df[col].value_counts(), "\n")
    
# Display Summary of Numerical Features
print("\n Numerical Features Summary: \n")
print(df[numerical_features].describe())