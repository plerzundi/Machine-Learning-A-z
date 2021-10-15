# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:43:49 2021

@author: plerzundi
@description: KMEANS MODEL
"""

# Librerias principales
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Importar Dataset
df = pd.read_csv("Mall_Customers.csv")

X = df.iloc[:,[3,4]].values

# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
        kmeans = KMeans(n_clusters = i, init='k-means++',max_iter=300,n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Método de codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar el método K-means para segmentar el data set
kmeans = KMeans(n_clusters = 5, init='k-means++', max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizacion de los clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s= 100, c="red", label = "Estandar")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s= 100, c="blue", label = "Descuidados")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s= 100, c="green", label = "Objetivo")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s= 100, c="cyan", label = "Conservadores")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s= 100, c="magenta", label = "Cautos")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s= 200,c = "black",label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales ( en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()