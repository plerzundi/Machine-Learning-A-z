# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 02:12:30 2021

@author: plerzundi
"""


# Importar librerias
import pandas as pd
import matplotlib.pyplot as plt

# Importación del dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_reward+=reward

# Visualizacion de los resultados
plt.hist(ads_selected)
plt.title("Histograma de seleccion de anuncios")
plt.xlabel("Anuncio")
plt.ylabel("Numero de veces que ha sido visualizado")
plt.show()
    