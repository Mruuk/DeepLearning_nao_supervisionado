# Construção da GAN

## criamos a variavel gan, e vamos utilizar o método simple_gan, que foi importado, passamos como parâmetro, o gerador, o discriminador, que são as duas redes neurais, e passamos um outro parâmetros, o normal_latent_sampling e passamos o parametros 100, o que quer dizer que vamos gerar 100 imagens

- em outras palavras será gerada 100 imagens do gerador e 100 imagens do discriminador(no caso, imagens da base de dados), pois a função do discriminador é comparar as imagens passadas para ele, ondes são elas as imagens da base e as imagens geradas pela rede neural geradora
- e perceba que esse valor deve ser igual ao input_dim da nossa rede geradora

```python
gan = simple_gan(gerador, discriminador, normal_latent_sampling((100,)))
```

### criamos o nosso modelo

- chamamos o adversarialmodel, já importado. e recebe os seguintes parâmetros:
  - base_model, que recebe nossa variavel gan
  - player_params, que recebe os pesos de cada uma das redes
    - primeiro pegamos do gerador e usamos o parametros trainable_weights, ele que é responsável por pegar os pesos da rede
    - em segundo pegamos os pesos da rede discriminador

```python
modelo = AdversarialModel(base_model = gan,
                          player_params= [gerador.trainable_weights,
                                          discriminador.trainable_weights])
```

### vamos agora configurar mais um parâmetro, adversarial_compile

- passamos como parâmetro:
  - o adversarial_optimizer, sendo nosso otimizador, que recebe o otimizador que importamos, Adversarial OptimizerSimultaneous
    - é o otimizador da rede completa, que envolve as duas redes
    - ele é responsável por atualizar cada uma das redes neurais, simultaneamente em cada um dos batchs
  - player_optimizer, que são os otimizadores de cada uma das redes, e passamos o adam para as duas
  - loss, recebe o binary_crossentropy

```python
modelo.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous,
                           player_optimizer = ['adam', 'adam'],
                           loss= 'binary_crossentropy')
```

### etapa de treinamento do nosso modelo

- o nosso fit recebe como parâmetros:
  - x, sendo nossos previsores_treinamento, nossa base de dados, que são as imagens real, onde queremos nos basear para geração
  - y, sendo o método gan_targets, com 60000, referente ao número de registros da base de dados
  - epochs, nosso número de épocas, importante rodar mais vezes, esse tipo de rede neural traz melhores resultados quando rodado mais épocas
  - batch_size, recebe 256

```python
modelo.fit(x = previsores_treinamento,y = gan_targets(60000), epochs = 100,
                                                      batch_size = 256)
```

### geramos nossa entrada de números random da nossa rede geradora

- 10 registros com 100 atributos
  - os 100 atributos referentes da camada de entrada do nosso gerador
- para gerar as imagens, criamos uma variavel previsao, recebendo nosso predict da rede gerador, e passamos a amostras, onde são nosso input_dim/camada de entrada

```python
amostras = np.random.normal(size = (10,100))
previsao = gerador.predict(amostras)
```

### para visualizar as imagens geradas

- previsao.shape[0], são nossos 10
- previsao[i, :], queremos pegar as 10 linhas e todas as colunas
- cmap ='gray', para a imagem ficar em tons de cinza
  - se tiver trabalhando com imagens coloridas, não é necessário essa configuração
- plt.show(), para efetivamente visualizar

```python
for i in range(previsao.shape[0]):
    plt.imshow(previsao[i, :], cmap = 'gray')
    plt.show()
```

### e foi gerada uma imagens aparti do zero
