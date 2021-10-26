# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 02:59:55 2021

@author: plerzundi
"""


# LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# IMPORTAR EL DATASET
dataset = pd.read_csv("Market_Basket_Optimisation.csv",header=None)

# PRE PROCESADO
transations = []
for i in range(0,len(dataset)):
    transations.append([str(dataset.values[i,j]) for j in range(0,20)])

# ENTRENAR ALGORITMO APRIORI
# 3*7/7500 = 0.0028
from apyori import apriori
rules = apriori(transations, min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

# Visualización de resultados
results = list(rules)
results[0]