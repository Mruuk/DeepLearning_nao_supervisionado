#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:08:53 2022

@author: lisboa
"""

from rbm import RBM
import numpy as np

rbm =  RBM(num_visible = 6, num_hidden = 3)

base = np.array([[0,1,1,1,0,1],
                  [1,1,0,1,1,1],
                  [0,1,0,1,0,1],
                  [0,1,1,1,0,1],
                  [1,1,0,1,0,1],
                  [1,1,0,1,1,1]])

filmes = ['Freddy x Jason', 'O Ultimato Bourne', 'Star Trek',
          'Exterminador do futuro', 'Norbit', 'Star Wars']

rbm.train(base, max_epochs = 5000)

rbm.weights

leonardo = np.array([[0,1,0,1,0,0]])

camada_escondida = rbm.run_visible(leonardo)
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(leonardo[0])):
    if leonardo[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
        

ruan = np.array([[0,0,1,1,0,1]])

camada_oculta = rbm.run_visible(ruan)
recomendacao2 = rbm.run_hidden(camada_oculta)

for i in range(len(ruan[0])):
    if ruan[0,i] == 0 and recomendacao2[0,i] == 1:
        print(filmes[i])
