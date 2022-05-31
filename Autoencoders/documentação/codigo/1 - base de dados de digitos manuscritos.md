# Implementação de um autoencoder bem simples

## importações necessárias

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:37:17 2022

@author: lisboa
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
```

## carregamento dos dados

- quando trabalhamos com algoritmos não supervisionados, não temos classe supervisora, então, onde nós colocariamos a classe, vamos apenas colocar, "_"(underline), isso indica que não teremos classe supervisora
- como se trada de uma codificação e decodificação, não precisamos colocar classe, pegaremos os dados de determinado digito e vamos fazer o encoding e depois o decoding e verificar se aparece igual, ou qual a diferença

```python
# carregar base de dados
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
```

## normalização dos dados

- como se trata de uma base de imagens, normalizamos em valores RGB, 255 por tanto
- transformamos também em float32, para conseguir fazer a divisão
- também pode utilizar o minmaxscaler

```python
# normalização
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255
```

## Redimensionamento dos previsores

- faremos um reshape, pois esse padrão que é carregado pelo keras, tem primeiramente a quantidade de registros e depois as dimensões, largura e altura. Como isso precisamos multiplicar esses dois valores
- Note que temos (28,28) o que se multiplicarmos teremos 784
- np.prod, é o produto, ele fará a multiplicação
- o len, retorna seu tamanho
- transformando apenas o (28,28) em 784, e mantendo o tamano de 60000 registros para o treinamento e 10000 para teste

```python
# redimensionamento 
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))
```
