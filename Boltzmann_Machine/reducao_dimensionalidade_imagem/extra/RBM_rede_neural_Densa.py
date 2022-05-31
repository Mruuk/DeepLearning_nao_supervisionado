#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:35:51 2022

@author: lisboa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline

# carregamento dos dados
base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target

# Normalização na escala de 0 e 1
normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

# Divisão da base para teste e treinamento
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.2, random_state=0)

# Criação e configuração da RBM
rbm = BernoulliRBM(random_state = 0)
rbm.n_iter = 25
rbm.n_components = 50
mlp_rbm = MLPClassifier(random_state=0,
                        max_iter = 1000,
                        batch_size = 50,
                        verbose = 1)
classificador_rbm = Pipeline(steps = [('rbm', rbm),('mlp',mlp_rbm)])
classificador_rbm.fit(previsores_train, classe_train)

plt.figure(figsize = (20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10,10, i +1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

previsoes_rbm = classificador_rbm.predict(previsores_test)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_test)

mlp_simples = MLPClassifier(random_state=0,
                        max_iter = 1000,
                        batch_size = 50,
                        verbose = 1)
mlp_simples.fit(previsores_train, classe_train)
previsoes_mlp = mlp_simples.predict(previsores_test)
precisao_mpl = metrics.accuracy_score(previsoes_mlp, classe_test)