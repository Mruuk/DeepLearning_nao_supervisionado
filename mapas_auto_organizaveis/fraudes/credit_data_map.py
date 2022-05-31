#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:28:06 2022

@author: lisboa
"""
# default == inadiplante, se foi ou não aprovado para crédito bancário
# o aprovou, 1 não aprovou

import pandas as pd
from minisom import MiniSom 
import numpy as np

base = pd.read_csv('recursos/credit_data.csv')
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

from sklearn.preprocessing import MinMaxScaler

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

# construção do mapa

som = MiniSom( x = 15, y = 15, input_len= 4, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
        markerfacecolor = 'None', markersize = 10,
        markeredgecolor = color[y[i]], markeredgewidth = 2)

# análise, buscando clientes outliers
mapeamento = som.win_map(X)
suspeitos = np.concatenate((mapeamento[(4,5)], mapeamento[(6,3)]), axis = 0)
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 4])
    
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]

df = pd.DataFrame(suspeitos_final)
df.to_csv('suspeitos_de_fraude.csv', index=False)
