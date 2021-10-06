# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:49:51 2021

@author: plerzundi
"""

# LIBRERIAS PRINCIPALES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar Datasets
dataset = pd.read_csv("Social_Network_Ads.csv")

# Dividir variable independiente vs dependiente
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Dividir el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el modelo de regresion logistica para el conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Representacion grafica de los resultados del algoritmo Conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Conjunto de entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo estimado')
plt.legend()
plt.show()

# Representacion grafica de los resultados del algoritmo Conjunto de testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Conjunto de Testing)')
plt.xlabel('Edad')
plt.ylabel('Sueldo estimado')
plt.legend()
plt.show()
