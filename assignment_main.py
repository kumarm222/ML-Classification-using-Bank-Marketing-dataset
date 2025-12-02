# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:22:02 2025

@author: Mayank
"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, auc
    )
    
# Loading dataset
df = pd.read_csv("../data/bank-full.csv", sep=';')

# Preview
print(df.head())
print(df.info())
print(df.describe())

# Understanding the data before modelling
print("Columns:", df.columns.tolist())
print("Missing values per column:")
print(df.isnull().sum())
print(df["y"].value_counts())

#Exploring numeric feature distributions

num_cols= ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

df[num_cols].hist(figsize=(12,8))
plt.tight_layout()
plt.show()

# Exploring categorical features (Countplots)

cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

plt.figure(figsize = (14,12))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3,3,i)
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Finding correlation (Numeric Only)

plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot= True, cmap='coolwarm')
plt.show()

# Encoding categorical variables (One-Hot Encoding)

df_encoded = pd.get_dummies(df, drop_first = True)

print(df_encoded.head())
print(df_encoded.shape)

# Scaling (Important for SVM & KNN Only)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop('y_yes', axis= 1))

#Train/Test Split


X = df_encoded.drop('y_yes', axis = 1)
y = df_encoded['y_yes']

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size= 0.2, random_state=42, stratify=y
)

# Modelling with SVM, Random Forest, KNN and Decision Tree 
#Scaling (for SVM & KNN only)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL 1: SVM (RBF Kernel)

svm_model = SVC(
    kernel = 'rbf',
    C = 1,
    gamma = 'scale',
    class_weight = 'balanced'
    )

svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# MODEL 2: Random Forest Classifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth= None,
    class_weight= 'balanced',
    random_state= 42
    )

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# MODEL 3: KNN Classifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

# MODEL 4: Decision Tree Classifier

dt_model = DecisionTreeClassifier(
    criterion= 'gini', 
    max_depth= None,
    class_weight= 'balanced',
    random_state= 42
    )

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Visalise Tree Depth
print("Tree depth:", dt_model.get_depth())
print("Number of leaves:", dt_model.get_n_leaves())

# Full Evaluation Section

print("SVM Results:")
print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

print("\nKNN Results:")
print(classification_report(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))

print("n\Decision Tree Results:")
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))

# Comparing the models & preparing the results section
from sklearn.metrics import classification_report, confusion_matrix

print("SVM Results:")
print(classification_report(y_test, svm_pred))

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))

print("\nKNN Results:")
print(classification_report(y_test, knn_pred))

print("\nDecision Tree Recision:")
print(classification_report(y_test, dt_pred))

# KMeans Clustering
X_cluster  = df_encoded.drop('y_yes', axis= 1)

# Scaling the data
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

#Finding the optimal number of cluster(elbow method)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia_values = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_cluster_scaled)
    inertia_values.append(kmeans.inertia_)
    
plt.plot(K_range, inertia_values, marker= 'o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for  Optimal K")
plt.show()

# Fit KMeans with K = 4
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

df['cluster']= clusters

# Understanding each cluster
#(Mean of numeric features only)
df.groupby('cluster').mean(numeric_only=True)

#(Job distribution within each cluster)
df.groupby('cluster')['job'].value_counts(normalize=True)

#(Education distribution within each cluster)
df.groupby('cluster')['education'].value_counts(normalize=True)

# Visualizing Clusters
# Cluster size visualization
sns.countplot(x='cluster', data= df)
plt.title("Cluster Sizes")
plt.show()

# Balance across clusters
sns.boxplot(x='cluster', y='balance', data =df)
plt.title("Balance Distribution per Cluster")
plt.show()

#Age across clusters
sns.boxplot(x='cluster', y='age', data = df)
plt.title("Age Distribution per Cluster")
plt.show()

# Cluster Interpretation
df.groupby('cluster').mean(numeric_only= True)

df.groupby('cluster')['job'].value_counts(normalize=True)

df.groupby('cluster')['education'].value_counts(normalize=True)

