# criação do autoencoder

## O fator de compactação

- divide o número de neurônios de entrada e o número de neurônios da camada oculta
- temos 24,5% de compactação das imagens

```python
fator_compactacao = 784/32
```

## criamos a rede neural densa

- nó visível de 784
- nó invisível de 32
- e nó de saída tem que ser o mesmo da entrada, 784
- funcão de ativação da camada oculta, relu, por padrão o que tem dados melhores resultados e com experimentações, é a função relu
- função de ativação da saída, passamos o sigmoid, que vai retornar entre o e 1, e só estamos usando sigmoid, pois normalizamos os registros entre 0 e 1
- é muito comum também, utilizar a tahn(tangente hiperbólica)
- note que começamos com 748 neurônios e fomos para 32 e por fim, retornamos para 784, exatamento como na lógica dos autoencoders

```python
autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
```

## Para visualizarmos os dados desse autoencoder, a sua estrutura

- 25120, parametros e o shape de 32. Temos 784 neurônios na camada de entrada que estão relacionados com 32 neurônios da camada oculta
-verificando se bate os valores com 25120

$784\times32=25088$

- vai adicionar a camada de bias
  - a camada de bias é adicionada internamente, não é necessário fazer essa configuração

$25088+32= 25120$

- agora temos 25872 parametros e o shape de 784, nesse caso temos 32 neurônios da camada oculta para cada 784 da camada de saída

$32*784=25088$

- mais as bias

$25088+784=25872$

- e se fizer o sumatório desses valores teremos, 50992 parametros que serão configurados, em outras palavras, serão os pesos entre os neurônios

```python
autoencoder.summary()
```

## faremos a compilação

- optimizer, será o adam
- loss function, será o binary_crossentropy
- metrics, será accuracy

```python
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
```

## e por fim, efetivamos o treinamento

- onde o método fit, necessita de dois parametros,  que por sua vez, são os previsores e a classe, porém como se trata de um algoritmo onde não temos classe, ou melhor nossa classe é o próprio previsor, por isso self supervised learning, ou seja, aprendizagem supervisionada por se própria. Então o segundo parametro, a classe, é o próprio previsor
- colocamos 50 épocas, mais o recomendado é mais, um algoritmo de autoencoder, para ter resultados satisfatórios, necessita de mais épocas
- batch_size de 256
- e por fim validation_data, para já fazer a validação em seguida com a base de dados teste, seguindo o mesmo princípio de comparar com ele mesmo

```python
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))
```
