#!/usr/bin/env python
# coding: utf-8

# In[2]:


#GredSearch ve Çoklu Nominal Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Veri kümesini yükle
df = pd.read_csv('C:\\Users\\abdul\\OneDrive\\Masaüstü\\diabetes.csv')

# Özellikleri (X) ve hedefi (y) ayır
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Veri kümesini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Çoklu Nominal Naive Bayes sınıflandırıcısını başlat
nb_classifier = MultinomialNB()

# Grid search için parametre grid'ini tanımla
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

# Çapraz doğrulama ile grid search yap
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi modeli ve hiperparametrelerini al
best_nb_classifier = grid_search.best_estimator_
best_alpha = grid_search.best_params_['alpha']

# En iyi modeli kullanarak test setinde tahminlerde bulun
y_train_pred = best_nb_classifier.predict(X_train)
y_test_pred = best_nb_classifier.predict(X_test)

# Doğruluğu hesapla
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print("En iyi alpha değeri:", best_alpha)
print("Eğitim doğruluğu:", accuracy_train)
print("Test doğruluğu:", accuracy_test)

print('Eğitim seti performansı:\n', classification_report(y_train, y_train_pred))
print('Test seti performansı:\n', classification_report(y_test, y_test_pred))


# In[ ]:




