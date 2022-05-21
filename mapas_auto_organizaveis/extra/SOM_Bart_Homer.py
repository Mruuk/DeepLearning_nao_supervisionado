#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:40:18 2022

@author: lisboa
"""

import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

base = pd.read_csv('recursos/personagens.csv')
base.insert(loc = 0, column = 'id',
            value = base.index)

X = base.iloc[:, 0:7].values
y = base.iloc[:, 7].values

normalizado = MinMaxScaler(feature_range=(0,1))
X = normalizado.fit_transform(X)


som = MiniSom(x = 9, y = 9, input_len = 7, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 500)

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']

y[y == 'Bart'] = 0
y[y == 'Homer'] = 1

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
        markerfacecolor = 'None', markersize = 10,
        markeredgecolor = color[y[i]], markeredgewidth = 2)
    
mapeamento = som.win_map(X)
suspeitos = mapeamento[(2,2)]
suspeitos = normalizado.inverse_transform(suspeitos)



classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 7])

classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final =  suspeitos_final[suspeitos_final[:, 7].argsort()]


        