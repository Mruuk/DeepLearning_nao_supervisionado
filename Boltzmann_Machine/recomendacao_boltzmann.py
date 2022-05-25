#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:55:30 2022

@author: lisboa
"""
# importação do algoritmo da rbm
from rbm import RBM
import numpy as np

# definindo nó visível e oculto
rbm = RBM(num_visible = 6, num_hidden = 2)

# criando base de dados
base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

filmes = ["A Bruxa", ' A Invocação do mal', 'O Chamado',
          'Se beber não case', 'Gente grande', 'American pie']

# realizando o treinamento
rbm.train(base, max_epochs = 5000)

#verificando pesos
rbm.weights

# nparray em formato de matriz
usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])

# visualizando qual neurônio especialista foi ativado para cada um dos usuários
rbm.run_visible(usuario1)
rbm.run_visible(usuario2)

# criando a recomendação
camada_escondida= np.array([[0,1]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
    if usuario1[0,i] == 0 and recomendacao[0,i] == 1:
        print('usuario 1: ', filmes[i])
        
camada_escondida= np.array([[1,0]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario2[0])):
    #print(usuario1[0,i])
    if usuario2[0,i] == 0 and recomendacao[0,i] == 1:
        print('usuario 2: ', filmes[i])