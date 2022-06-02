# Deep autoencoder

## Também chamada de stacked autoencoder, tem uma tendência de dar resultados bem melhores

### importações necessárias

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:25:51 2022

@author: lisboa
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
```

### mesmo processo já realizado nos exemplos anteriores

- carregamos os dados
- normalizamos os previsores
- redimensionamos os previsores

```python
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype("float32") / 255
previsores_teste = previsores_teste.astype("float32") / 255

previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))
```

### criação da deep autoencoder

- 784 - 128 - 64 - 32 - 64 - 128 - 784
- não tem muito segredo, a criação é de uma deep learning
- chamamos um summary apenas para verificar a estrutura
- compilamos e treinamos
- separamos em encode e decode para ficar visualmente fácil de entender e interpretar
- note que temos um autoencoder bem robusto com 222,384 parametros, que correspondem aos pesos que serão atualizados

```python

autoencoder = Sequential()

# Encode
autoencoder.add(Dense(units = 128, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 64, activation = 'relu'))
autoencoder.add(Dense(units = 32, activation = 'relu'))

# Decode
autoencoder.add(Dense(units = 64, activation = 'relu'))
autoencoder.add(Dense(units = 128, activation = 'relu'))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))

autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))
```

### agora vamos criar um modelo para capturar apenas a aprendizagem da rede, que corresponde ao codificador

- Input é uma camada própria do keras
- agora o nosso encoder possui mais de uma camada, diferente dos exemplos anteriores, então é necessário pegar cada uma delas
- cada camada encoder recebe uma layer, iremos ate a camada onde redimenciona para 32 pixels ou neurônios
- Model, é uma classe que se utiliza quando se quer fazer a criação manual de uma rede neural
  - entao instanciamos essa classe em nossa variavel encoder, e passamos os parametros necessários
  - a dimensao original, que é nosso primeiro parâmetro, que indica a entrada
  - em sequencia, precisamos colocar uma camada dentro da outra, e colocamos da camada mais interna para a mais externa
  - essa estrutura demostra que uma camada depende da outra
  - e dentro da camada encoder1, passamos a dimensao original
- chamamos novamente o método summary para avaliarmos a estrutura
- agora temos um novo modele, chamado encoder, e caso queira salvar esse encoder, para usar em uma outra máquina, e passar imagens como parametro e reduzir a dimensionalidade

```python
dimensao_original = Input(shape=(784,))
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
camada_encoder3 = autoencoder.layers[2]
encoder = Model(dimensao_original,
                camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))
encoder.summary()
```

### criamos as imagens codificadas e passamos o método predict, ja incluimos o decode também

```python
imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)
```

### Agora vamos visualizar dessas imagens

- e podemos verificar que tem mais nitidêz e mais preciso

```python
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    #print(i)
    #print(indice_imagem)
    
    # imagem original
    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10,10,i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
    # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
```
