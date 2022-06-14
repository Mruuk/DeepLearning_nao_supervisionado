# Implementação da rede do tipo GAN

## vamos fazer a geração automática de dígitos  manuscritos, utilizando a base mnist. Então a rede neural vai aprender a gerar dígitos manuscritos e depois vamos poder gerar dígitos novos, baseados no aprendizado da rede

### importações

- numpy
- matplotlib
- Sequential
- camadas
- um regularizador do keras, L1L2
  - entamos, vamos trabalhar com um conceito de regularização, que é importante para criamos esse tipo de rede neural, pois se não utilizarmos essa regularização, ele não vai dar resultados muito interessantes.
  - Essa função de regularização adiciona uma penalidade na aprendizagem, para evitar overfitting, quando faz o cálculo do erro ele tem uma função adicional onde vai adicionando uma penalidade se ele tiver um resultado muito baixo, assim envitando oproblema do overfitting.
  - Esse tipo de técnica é utilizado quando se tem muitas características, muitos atributos na base de dados, isso acontece principalmente em imagens pois imagens, temos um conjunto grande de pixels, principalmente se tivermos uma resolução muito grande, então é importante utilizar essa técnica
- e importamos o keras_adversarial, pois por padrão ele não possui nativamente uma biblioteca ou funções, onde possibilitam trabalhar com redes adversarias generativas, utilizaremos:
  - [keras-adversarial](https://github.com/bstriner/keras-adversarial), já possui as implementações prontas, e não é necessário fazer programação manual, para fazer a ligação entre as duas redes neurais, a rede que gera as imagens e a discriminadora
  - Classes: adversarialmodel, simple_gan, gan_targets, adeversarialoptimizerSimultaneous, normal_latent_sampling

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:18:49 2022

@author: lisboa
"""

import  numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Reshape
from keras.regularizers import L1L2
import keras_adversarial
```

### carregamento da base de dados

- não vamos buscar a classe
- nesse método load que vamos trabalhar, ele vai puxzar a base de dados de teste, mas não vamos utilizar, nosso objetivo é utilizar as imagens **previsores_treinamento** para que a rede neural aprenda e possamo gerar novas imagens
- e normalizamos

```python
(previsores_treinamento,_), (_,_) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
```
