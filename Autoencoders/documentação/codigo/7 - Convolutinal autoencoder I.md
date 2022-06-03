# convolutinal autoencoder

## importações necessárias

### layers necessárias para uma rede neural convolucional, e como estamos falando de autoencoders, precisamos também do UpSampling2D e Reshape

- UpSampling2D, funciona como o inverso do MaxPooling2D, o maxpooling vai analizar e pegar o maior valor e por fim reduz a dimensionalidade, já o UpSampling é como se tivesse o resultado do maxpooling e ele vai retornar ao estado original
- Reshape, vai redimensionar o vetor gerado pelo flatten, novamente em matriz

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:41:53 2022

@author: lisboa
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
```

### carregamos a base de dados

```python
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
```

### faremos o redimensionamento, com o reshape, semelhante aos anteriores porem, com uma mudança, pois precisamos colocar no padrão para conseguirmos fazer a aplicação das redes neurais convolucionais

- passando as dimensões, o shape que era 784, agr mantemos 28, 28, e acrescentamos o número de canais, e colocamos 1 pois a imagem esta em escala de cinza, basta apenas 1 canal

```python
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), 28, 28, 1))
previsores_teste = previsores_teste.reshape((len(previsores_teste), 28, 28, 1))
```

### normalização

```python
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255
```

### criação do autoencoder

- começamos com uma camada convolucional, com 16 filtros/kernels, kernel_size seria o tamanho do nosso detector de caracteristicas, nesse caso, ele possui 3 linhas e 3 colunas, função de ativação para redes convolucionais, sempre utilizamos a relu, onde a ideia é eliminar os pixels negativos, e passamos o shape de entrada
- criamos uma camada de maxpooling, com o parametro do pool size (2,2), ela define o tamanho da janela, que percorre nossa matriz, buscando os maiores valores e redimensionando, com a representação desses valores
- adicionamos mais uma camada de convolução e de maxpooling, mudando apenas alguns parametros
  - valor dos filtros agora serão de 8
  - e vamos adicionar um padding = 'same', responsável por indicar como que a imagem vai ser passada, qual o formato da imagem, quando colocamos o 'same', dizemos que primeiro colocamos o batch, a largura e altura da imagem e o número de canais. Caso não coloquemos esse parâmetro nosso código dará erro
- faremos mais uma camada de convolução, com o strides = (2,2), ele define de quantos em quantos pixels a imagem deve andar, nesse caso 2 em 2, por padrão é 1 em 1
- chamamos um flatten
  - perceba que não há obrigatóriedade de colocar uma camada maxpooling em sequencia
- terminamos com um array nas dimensões (4,4,8), com o flatten transformando em um vetor de 128 posições
- Voltamos para as dimenções 4,4,8 para que possa decodificar a imagem, e usamos o Reshape para isso
- até o flatten não fizemos o encoder

```python
autoencoder = Sequential()

# encoder
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu',
                       input_shape = (28,28,1)))
autoencoder.add(MaxPooling2D(pool_size= (2,2)))

autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',padding = 'same'))
autoencoder.add(MaxPooling2D(pool_size= (2,2), padding = 'same'))
# 4, 4, 8
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',padding = 'same', strides = (2,2)))

autoencoder.add(Flatten())

autoencoder.add(Reshape((4,4,8)))
```

![convolutional](/img/convolutional_encoder.png)
