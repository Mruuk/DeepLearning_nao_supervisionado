# Tipos de autoencoders

## Uma coisa que se pode fazer quando falamos de autencoders, é aumentar o número de neurônios da camada oculta

- O autoencoders pode ser utilizado para reduzir a dimensionalidade
  - tem-se 50 entradas é deseja diminuir parar 20 entradas
  - mais pode-se fazer o contrário, caso queira ter mais atributos na base de dados, queira ter uma representação mais precisa, ou mais detelhada dos seus dados
- neste exemplo temos 3 registros na camada de entrada, importante salientar que o número de entradas é igual ao número de saídas, pois fazemos primeiro o processo de codificação e depois de decodificação. As 3 entradas serão transformadas em 5 neurônios, ou seja, 5 valores diferentes, o que vai aumentar um pouco a quantidade de dados

![aumentar a camada](/Autoencoders/img/aumentar_oculta.png)

- Porém quando se usa essa técnica de colocar mais neurônios na camada oculta, ele pode dar um certo problema, pois ele pode simplesmente copiar os valores da camada de entrada, diretamente para a camada oculta. Então ele nao vai ter muito poder de discriminação
  - e para isso foram criados outros tipos de autoencoders, para solucionar esse problema, de quando quer fazer esse processo inverso

## O primeiro nós temos o sparse autoencoder

### Na representação 2 entradas vão virar 4 neurônios, 2 saídas

- Um dos mais populares autoencoders, é um dos mais básicos que se vai encontrar
- Usa uma técnica de regularização para previnir overfitting, previnir que se adapte demais ao dados, que é o que pode acontecer quando tenho tenho maior quantidades de neurônios na camada escondida
- Ele não vai usar todos os neurônios da camada oculta, ele vai colocar apenas valores pequenos nos pesos

   ![sparse autoencoder](/Autoencoders/img/sparse.png)

### ele vai desconsiderando os neurônios para cada registro, isso de forma aleatória, assim evitando que ele se adapte demais à base de dados

   ![desconsidera](/Autoencoders/img/descosiderar.png)

## Denoising autoencoder

### Temos as mesmas quantidade de entrada, oculta e saída

- Ele vai fazer, é modificar os valores da camada de entrada, alterando alguns neurônios para valor zero
- o simbolo do trinagulo, seginifica edição, mostrando que os neurônios vão sofre edições, onde serão zerados, para cada registro, será zerada algumas entrada, para evitar o processo do overfitting
- Quando os pesos são atualizados, a camada de saída é comparada com os valores  originais para obter o valor do erro
- Como na ideia de uma rede neural clássica, numa apredizagem supervisionada, onde terá uma classe, e la no final será feita a comparação da classe que fez a previsão com a classe que esta la na base de dados. No nosso caso aqui, vamos comparara os valores de saída com os valores de entrada

![denoising](/Autoencoders/img/denoising.png)

## Contractive autoencoder

- Adicina uma função de penalidade quando os pesos são atualizados
- Ao chegar na camada de saída com os valores, é feita o backpropagation, fazendo o calculo do erro e terá umas funções adicionais, para essa função de calcular o erro. Ou seja, ele vai colocar um valor de penalidade, que ele vai ter a tendência de dar melhor os dados

![contractive](/Autoencoders/img/contractive.png)

## Deep autoendoder

- Temos a camada de entrada e temos em mesma quantidade a camada de saída, mas perceba que temos camadas de encoding e camadas de decoding, reduzindo muito a dimensionalidade em a cada camada oculta
- Compressed feature vector, o vetor de caracteristicas comprimido, onde temos a codificação final
- Deep autoencoder, também é chamado de stacked autoencoder

![deep](/Autoencoders/img/deep.png)

## Convolutional autoencoder

### Baseado em redes enurais convolucionais

- temos a imagem original, e o processo, segue com camada de convolução sucessivas, para reduzirmos sua dimensionalidade, a pois temos o flatten, o vetor de caracteristicas, e depois temos o encoder, e por fim teremos os reshapes, fazendo então a decodificação e desconvolusionando as camadas

![convolutional](/Autoencoders/img/convolutional.png)

## essa área de autoencoder é bastante vasta, então esses são alguns tipos de autoencoders que possuimos, se for pesquisar, achará muito mais deles
