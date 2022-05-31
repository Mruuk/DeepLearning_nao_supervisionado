# Autoencoder e classificação

## vamos trabalhar com o autoencoder juntamente com a técnica de classificação

- o  autoencoder, é uma técnica de aprendizagem não supervisionada, que é utilizada para fazer redução de dimensionalidade
- programaremos o mesmo autoencoder do exemplo anterior, porém faremos a aplicação de uma classificação e iremos comparar os resultados, entre usar a base de dados com ou sem redução de dimensionalidade

### importações, com um extra do np_utils, para criamos uma variável do tipo dummy

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:10:10 2022

@author: lisboa
"""


import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils
```

### criamos os previsores treinamento e teste, e como vamos fazer a avaliação de como um algoritmo de classificação, vai se comportar, juntamente com a aplicação do autoencoder, entao vamos criar também a classe treinamento e teste

- normalizamos os previsores
- transformamos as classes em classes dummy
  - como temos 10 valores, do 0-9, faremos o dummy para que tenhamos 10 colunas, para passar como parâmetros, para que não gere erro

```python
(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255
classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento)
classe_dummy_teste = np_utils.to_categorical(classe_teste)
```

### colocamos no formato de 784 ao invés (28, 28), multiplicando 28 por 28

- a cada linha dos previsores é o que valente a uma imagem

```python
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))
```

### Mesma estrutura do autoencoder do exemplo anterior

```python
# 784 - 32 - 784
autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
autoencoder.compile(optimizer = 'adam', loss= 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))

dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
```

### vamos redimensionar os previsores de 784 para 32 pixels

- O que vamos fazer é compararmos essas o resultado utilizando a base de dados completa, com 784 pixels e a base de daods codificada, com somente 32 pixels
- faremos uma rede neural densa para fazer a previsao dos previsores com dimensionalidade de 784
- faremos outra rede neural densa para fazer a previsao dos previsores com a dimensionalidade de 32

```python
previsores_treinamento_codificados = encoder.predict(previsores_treinamento)
previsores_teste_codificados = encoder.predict(previsores_teste)
```
