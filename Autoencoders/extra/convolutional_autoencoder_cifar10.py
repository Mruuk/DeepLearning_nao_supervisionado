#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:42:55 2022

@author: lisboa
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, UpSampling2D, Reshape

(previsores_treinamento, _), (previsores_teste, _) = cifar10.load_data()

#previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
#previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))

previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filters = 32,kernel_size = (3,3), activation = 'relu',
                       input_shape = (32, 32, 3)))
autoencoder.add(MaxPool2D(pool_size= (2,2)))

autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', padding = 'same'))
autoencoder.add(MaxPool2D(pool_size= (2,2), padding = 'same'))

autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', padding = 'same', strides = (2,2)))

autoencoder.add(Flatten())

autoencoder.add(Reshape((4,4,16)))

# Decode
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu',
                       padding= 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu',
                       padding= 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 10, kernel_size = (3,3), activation = 'softmax', padding = 'same'))
autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss='categorical_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 100, batch_size = 256,
                )