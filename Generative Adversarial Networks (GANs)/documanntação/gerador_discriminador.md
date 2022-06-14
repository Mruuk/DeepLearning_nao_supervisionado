# Gerador e discriminador

## vamos impementar as duas redes neurais, a geradora e a discriminadora

### iniciamos com a geradora

- modelo sequencial, do tipo denso
- camada de entrada 100 neurônios e camadas ocultas 500, esse valores, não seguiram a lógica de somar entrada e saída e dividir por 2, mas foram parâmetros tirado de um turing, nem sempre o método de somar e dividir trará bons resultados para uma rede neural.
- função de arivação, relu
- kernel_regularizer, utilizamos o L1L2, cada um deles com 0,000001,(1e-5), valor esse recomendado pela documentação do keras
  - e basicamente, temos dois regularizador, o L1 e o L2, e podesse utilizar esses dois regularizadores juntos que ele irá aplicar duas fómulas matemáticas, onde ele vai adicionar penalidades na apredizagem, para evitar o overfitting
- camada de saída, utilizamos 784 neurônios pois, como estamos trabalhando com imagens, precisamos formar uma na camada de saída para que a rede neural discriminadora possa avalia-la, pensando nisso, nossas imagens tem um tamanho de 28x28, logo sua multiplicação nos trará 784.
  - função de ativação, utilizamos sigmoid, pois como normalizamos nossa base da dados para valores entre 0 e 1 dos pixels, então precisamos fazer o mesmo aqui, e a função fará exatamente isso.
- Reshape, utilizamos o Reshape pois, como a saída de uma rede neural é gerada um vetor, e precisamos passar isso para a segunda rede neural como uma imagem, então dimensionalizamos ela para 28,28, novamente, note que estamos entregando a mesma dimensão de anteriormente e que será a mesma dimensão das imagens que nossa rede discriminadora onde fará a comparação com as imagens geradas pelo rede geradora

```python

# Gerador
gerador = Sequential()

gerador.add(Dense(units= 500, input_dim= 100, activation=  'relu',
                  kernel_regularizer=L1L2(1e-5, 1e5)))
gerador.add(Dense(units= 500, activation=  'relu',
                  kernel_regularizer=L1L2(1e-5, 1e5)))
gerador.add(Dense(units= 784, activation= 'sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)))
gerador.add(Reshape((28,28)))
```

### Criamos uma rede discriminadora, sendo seu modelo sequencial e do tipo denso também

- primeira camada é um InputLayer com seu parâmetro o input_shape, onde passamos a dimensão das imagens que irá receber nossa rede neural.
  - pasasmos esse inputlayer, pois ele vai receber imagens do tipo 28x28
- passamos o método flatten, onde convete os dados para vetores, assim como é feito em redes convolucionais.
- e note que até o momento as duas camadas geradas, são como se fosse um pré-processamento dos dados
- criamos então nossa camada densa com 500 neurônios e sua função relu, passamos também o kernel_regularizer com mesmos parametros da rede neural anterior
- e nossa camada de saida vai receber apenas 1 neurônio, pois ela vai gerar uma probabilidade, onde foi visto na etápa de teoria, que nossa rede discriminadora irá comparar as imagens geradas com as imagens do banco de dados e avaliará, caso o valor retorne 1, então temos uma imagem gerada com sucesso, caso contrário, falhamos em gera-la, e devido a isso, utilizaremos a função sigmoid

```python
# Discriminador
discriminador = Sequential()
discriminador.add(InputLayer(input_shape=(28,28)))
discriminador.add(Flatten())
discriminador.add(Dense(units= 500, activation= 'relu', 
                        kernel_regularizer=L1L2(1e-5,1e-5)))
discriminador.add(Dense(units= 500, activation= 'relu', 
                        kernel_regularizer=L1L2(1e-5,1e-5)))
discriminador.add(Dense(units= 1, activation= 'sigmoid',
                        kernel_regularizer=L1L2(1e-5, 1e-5)))
```
