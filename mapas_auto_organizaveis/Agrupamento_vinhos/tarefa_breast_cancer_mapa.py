#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:38:06 2022

@author: lisboa
"""

import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base1 = pd.read_csv('recursos/entradas_breast.csv')
base2 = pd.read_csv('recursos/saidas_breast.csv')

X = base1.iloc[:, 0:30].values
y = base2.iloc[:,0].values

normalizador = MinMaxScaler(feature_range=(0,1))

X = normalizador.fit_transform(X)

som = MiniSom(x= 10, y= 10, input_len=30, sigma = 3.0, learning_rate= 0.5, random_seed= 0)

som.random_weights_init(X)

som.train_random(data= X, num_iteration= 1000)

som._weights
som._activation_map
q = som.activation_response(X)

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)
    

