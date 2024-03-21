#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

# Veri setini yükleme
df = pd.read_csv("C:\\Users\\abdul\\OneDrive\\Masaüstü\\diabetes.csv")

# Eksik veri kontrolü ve düzeltme
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']].replace(0,np.NaN)
df.fillna(df.mean(), inplace=True)

# Veri setini normalize etme ve hedef değişkeni ayırma
X = df.drop(columns="Outcome")
y = df["Outcome"]

# Min-max normalization
for column in X.columns:
    if column != "DiabetesPedigreeFunction":
        X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Gaussian Naive Bayes sınıflandırıcısını başlatma ve eğitme
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Randomized Search için parametre dağılımı
param_dist = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

# Randomized Search
random_search = RandomizedSearchCV(estimator=gnb, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# En iyi modeli ve hiperparametrelerini al
best_nb_classifier = random_search.best_estimator_

# Test seti üzerinde tahmin yapma
y_pred = best_nb_classifier.predict(X_test)

# Doğruluk ve sınıflandırma raporu
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix'i görselleştirme
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
cm_display.plot()


# In[ ]:




