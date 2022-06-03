# Finalização do autoencoder convolucional

## vamos realizar a segunda etapa do autoencoder, etapa essa que vem depois do flatten, referente ao decoder das imagens

- UpSampling2D é como o MaxPooling2D, mas ao invés de diminuir a dimensionalidade vamos aumenta-la com o upsampling
- seguimos aqui com a mesma ideia da abordagem anterior, começa com o valor maior, ai vai diminuindo, depois vai aumentando ate atigindo o valor total/inicial
- faremos o processo inverso da etapa anterior, vamos aumentar as camadas
- e possuimos uma camada de saída, vamos usar apenas 1 filtro, o próprio algoritmo que vai determinar qual o melhor filtro será utilizado, e ele irá gerar uma imagens, passamos a função de ativação sigmoid, que vai retornar entre 0 e 1

```python
# Decoder
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',padding = 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',padding = 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', padding = 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 1, kernel_size = (3,3), activation = 'sigmoid',padding = 'same'))
autoencoder.summary()
```

### output

```python
In [11]: autoencoder.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 16)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 8)         1160      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 7, 7, 8)          0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 8)           584       
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 reshape (Reshape)           (None, 4, 4, 8)           0         
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 8)           584       
                                                                 
 up_sampling2d (UpSampling2D  (None, 8, 8, 8)          0         
 )                                                               
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 8)           584       
                                                                 
 up_sampling2d_1 (UpSampling  (None, 16, 16, 8)        0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 16, 16, 16)        1168      
                                                                 
 up_sampling2d_2 (UpSampling  (None, 32, 32, 16)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 32, 32, 1)         145       
                                                                 
=================================================================
Total params: 4,385
Trainable params: 4,385
Non-trainable params: 0
```

- em nossa ultima camada é necessário não colocar o parametro padding, para que possamos atingir (28, 28,1), ao invés de (32,32,1), como nossa dimensão é (28,28,1), basta tirar o padding = 'same', assim como para o primeiro parametro não precisamos dele, para o último também não, lembrando que apois ele temos a camada de saida, o processo de decode termina aqui

```python
# Decoder
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',padding = 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',padding = 'same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 1, kernel_size = (3,3), activation = 'sigmoid',padding = 'same'))
autoencoder.summary()
```

### output 2

```python
In [11]: autoencoder.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 16)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 8)         1160      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 7, 7, 8)          0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 8)           584       
                                                                 
 flatten_1 (Flatten)           (None, 128)               0         
                                                                 
 reshape (Reshape)           (None, 4, 4, 8)           0         
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 8)           584       
                                                                 
 up_sampling2d (UpSampling2D  (None, 8, 8, 8)          0         
 )                                                               
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 8)           584       
                                                                 
 up_sampling2d_1 (UpSampling  (None, 16, 16, 8)        0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 14, 14, 16)        1168      
                                                                 
 up_sampling2d_2 (UpSampling  (None, 28, 28, 16)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 28, 28, 1)         145       
                                                                 
=================================================================
Total params: 4,385
Trainable params: 4,385
Non-trainable params: 0
```

### realizamos o treinamento

- é um processo bem mais lento de treinamento comparado aos outros autoencoders, pois ele tem muito mais processamento, quando trabalhamos com as rede neurais convolucionais

```python

autoencoder.compile(optimizer = 'adam',loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))
```

### criamos o nosso encoder, pegando as camadas responsáveis por essa função

- vamos o método get_layer, e passamos o nome da camada, consultada no summary do autoencoder

```python
encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('flatten_1').output)
encoder.summary()
```

### Output

```python
In [13]: encoder.summary()

Output from spyder call 'get_cwd':
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_7_input (InputLayer)  [(None, 28, 28, 1)]      0         
                                                                 
 conv2d_7 (Conv2D)           (None, 26, 26, 16)        160       
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 13, 13, 16)       0         
 2D)                                                             
                                                                 
 conv2d_8 (Conv2D)           (None, 13, 13, 8)         1160      
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 8)          0         
 2D)                                                             
                                                                 
 conv2d_9 (Conv2D)           (None, 4, 4, 8)           584       
                                                                 
 flatten_1 (Flatten)         (None, 128)               0         
                                                                 
=================================================================
Total params: 1,904
Trainable params: 1,904
Non-trainable params: 0
```

### agora tendo essa variavel vamos criar as imagens codificadas e decodificadas passando o predict

```python
imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)
```

### e visualizamos as imagens, puxamos 10 imagens random e plotamos os originais, os codados e os encodados

- precisamos alterar apenas a dimensão das imagens codificadas, onde o reshape era de (8,4), agora será (16,8)
- se multiplicamos o 16,8 teremos o valor de 128, mesmo valor da nossa camada flatten, camada final do encoder

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
    plt.imshow(imagens_codificadas[indice_imagem].reshape(16,8))
    plt.xticks(())
    plt.yticks(())
    
    # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
```

### note que como fizemos um treinamento com poucas épocas, nosse algoritimo não foi tão preciso assim, mas realizando um treinamento com pelo menos 100 épocas, já trará resultados bem melhores, sendo até mais interessante que as abordagens dos casos anteriores, porém demanda de um poder computacional bem maior do que as abordagens dos casos anteriores
