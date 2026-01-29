# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 20:55:20 2026

@author: Lehlohonolo Saohatse
"""

'''

1. Applying min max scaling and standardization 
to a dataset using sckit-learn.

2. Observe the effects of scaling on model performance by
training a k-nn classifier before and after scaling

'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Iris dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset information
print("Dataset Info:")
print(x.describe())
print("\n Target Classes: ", data.target_names)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Train K-NN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict and Evaluate
y_pred = knn.predict(x_test)
print("Accuracy Without Scaling: ", accuracy_score(y_test, y_pred))

# Apply Min-Max Scaling
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Split scaled data
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(x_scaled,y,test_size=0.2, random_state=42)

# Train K-NN classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(x_train_scaled, y_train_scaled)

# Predict and Evaluate
y_pred_scaled = knn_scaled.predict(x_test_scaled)
print("Accuracy With Min-Max Scaling: ", accuracy_score(y_test_scaled, y_pred_scaled))

# Apply Standardization
scaler = StandardScaler()
x_stand = scaler.fit_transform(x)

# Split standardized data
x_train_std, x_test_std, y_train_std, y_test_std = train_test_split(x_stand, y, test_size=0.2, random_state = 42)

# Train K-NN classifier on Standardized data
knn_stand = KNeighborsClassifier(n_neighbors=5)
knn_stand.fit(x_train_std, y_train_std)
 
 # Predict and Evaluate
y_pred_std = knn_stand.predict(x_test_std)
print("Accuracy With Standardization: ", accuracy_score(y_test_std, y_pred_std))










