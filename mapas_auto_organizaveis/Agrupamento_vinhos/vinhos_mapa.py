#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:10:10 2022

@author: lisboa
"""

from minisom import MiniSom
import pandas as pd

base = pd.read_csv('recursos/wines.csv')
X = base.iloc[:,1:14].values # previsores
y = base.iloc[:,0].values # classe

# normalização
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

# construir o mapa auto organizavel
som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 0) 

# inicialização dos pesos
som.random_weights_init(X)

# treinamento
som.train_random(data = X, num_iteration = 100)

som._weights
som._activation_map
q = som.activation_response(X)

# visualizar resultados
from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T) # MID - mean inter neuron distance
colorbar()

#visualização dos resultados
# verificar como colocar cada um dos registros, no grafico MID
w = som.winner(X[0]) # Qual é o neurônio ganhador de cada registro, BMU
markers = ['o','s', 'D']
color = ['r', 'g', 'b']

# transformação dos valores da classe(y) de (1,2,3) em (0,1,2)
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

for i, x in enumerate(X):
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)