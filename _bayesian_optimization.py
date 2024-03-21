#!/usr/bin/env python
# coding: utf-8

# In[2]:


#min-max normalizasyonu ve bayesian optimization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from skopt import BayesSearchCV
from skopt.space import Real

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

# Naive Bayes sınıflandırıcısını başlatma
gnb = GaussianNB()

# Hiperparametre aralıklarını belirleme
param_dist = {'var_smoothing': Real(1e-9, 1e+1, prior='log-uniform')}

# Bayesian Optimization
opt = BayesSearchCV(
    estimator=gnb,
    search_spaces=param_dist,
    n_iter=50,
    cv=5
)

# Modeli eğitme
opt.fit(X_train, y_train)

# En iyi modeli seçme
best_nb_classifier = opt.best_estimator_

# Test seti üzerinde tahmin yapma
y_pred = best_nb_classifier.predict(X_test)

# Doğruluk ve sınıflandırma raporu
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix'i görselleştirme
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=best_nb_classifier.classes_)
cm_display.plot()


# In[ ]:




