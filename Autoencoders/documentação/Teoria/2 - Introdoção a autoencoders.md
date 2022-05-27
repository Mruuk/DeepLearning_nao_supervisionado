# Autoencoders

## possui camada de entrada, camada oculta e de saida

### O que acontece nesse processo, é que conforme o nome já nos diz, autoencoder, o encoder vem de condificação, semelhando a ideia de criptografia, e o auto por sua vez, significa que o algoritmo vai se codificar automáticamente. Passa uma base de dados e ele vai codificar automáticamente

![img](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/autoencoders.png)

### Vai passar a base de dados e ele vai codificar a base de dados, da camada de entrada para a oculta, a apois isso ele fará a decodificação para a camada de saída

- baseicamente tendo 5 atributos, ele fará alguns cálculos, temos também aquele conceito de pesos, ele vai transformar, os 5 atributos em somente 2 atributos e depois se quiser verificar qual é o registro original, utilizo os dois neurônios, no qual estão codificados, e faço a decodificação e temos o valor original

- se assemelha a um processo de criptografia, onde criptografa os dados em um formato onde não consegue identificar e depois utiliza o algoritimo para fazer o processo inverso, e assim vai conseguir descriptografar os dados, para visualizar eles originalmente

- notasse que é uma técnica que a principal função dela, a mais utilizada, é para reduzir a dimensionalidade do dados, principalmente quando for trabalhar com imagens que tem um grande número de pixels

- Conceito onde ele flui da esquerda para direita, onde difere da Boltzmann Machine, onde flui para todas as direções. Então temos um processo muito parecido com as redes neurais clássicas

- Então é um processo onde, será feita a alimentação dos dados da camada de entrada para a camada oculta, e da camada oculta para a de saída, onde utilizamos o algoritmo backpropagation, para atualizar os pesos. Então segue um processo muito parecido a de uma rede neural tradicional, do tipo feedForward, onde segue o fluxo, entrada oculta e saída, calcula o erro e retorna para entrada, repetindo assim o fluxo

![codificacao](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/codificacao.png)

## O Autoencoder, é considerado uma técnica de aprendizagem não supervisionada, porém é chamado de Self supervised learning

- Aprendizagem supervisionada por se próprio, ou seja, ele vai aprender a classificar, por exemplo, os registro de acordo com as entradas, então, ele mesmo vai aprender a codificar e decodificar

- É como se o próprio registro fosse uma classe. Ele vai passar determidado registro, vai fazer a codificação e depois vais ser realizada a decodificação

## Como funciona

### Importante salientar que a mesma quantidade de registros na camada de entrada deve ser para camada de saída, já que estamos codificando para 2 registros, neste caso, e ao decodificar ele deve retornar ao número original de resgistros da camada de entrada

- se tivermos 100 entradas, obrigatóriamente teremos 100 saídas

![comofunciona](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/comofunciona.png)

## Vamos considerar, cada um dos neurônios de entrada, como um atributo, vamos trabalhar neste exemplo apenas com 0 e 1, e consideremos uma base de dados de pessoas, onde idicaremos, se a pessoa tem ou não emprego, 1 para ter emprego e 0 para não tê-lo

- note que temos linhas diferentes ligando os neurônios
  - linhas sólida: +1
  - linhas pontilhadas: -1
  - valores meramente aleatórios, apenar para fins didáticos, pois o algoritmo quando executar, irá definir valores aleatórios, como se fosse os pesos, para cada uma das conexões

![ex1](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/ex1.png)

## Assumimos que esta pessoa não tem emprego, nem filhos, ela é casada e não tem carro

- e agora precisamos calcular os valores dos neurônios da camada oculta
- o objetivo é, pegar os dados, codificar num formato somente com 2 neurônios, para reduzirmos essa dimensionalidade
- para isso vamos calcular para cada um dos neurônios da camada oculta

### Primeiro neurônio da camada oculta

$(0\times1) + (0\times-1) + (1\times1) + (0\times-1) = 1$

- temos essa fórmula onde vai ter 4 multiplicações, e são 4 multiplicações por termos 4 neurônios na cada de entrada e ele recebe os valores dos neurônios da camada de entrada
- e onde temos a linha sólida, multiplicamos pelo valor 1, e onde for linha pontilhada, multiplicamos por -1

![ex1](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/ex2.png)

### Segundo neurônio da camada oculta

$(0\times-1) + (0\times-1) + (1\times1) + (0\times1) = 1$

![ex1](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/ex3.png)

### Esses valores, 1 e 1, é uma representação apenas com 2 dimensões, do registro (0,0,1,0), esse é o processo de codificação

## decodificação

- varemos com uma fórmula parecida

### Primeiro neurônio da camada de saída

$(1\times1) + (1\times-1) = 0$

### Segundo neurônio da camada de saída

$(1\times-1) + (1\times1) = 0$

### Terceiro neurônio da camada de saída

$(1\times1) + (1\times1) = 2$

### Quarto neurônio da camada de saída

$(1\times-1) + (1\times-1) = -2$

## Temos essa pré-decodificação

![predecod](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/predecod.png)

## Agora iremos aplicar uma função de ativação, softmax

- irá transforma em 1 valores maiores ou igual a 0 e vai transformar em 0 ous outros valores
- fazendo isso, note que o registro de saída é o mesmo do da entrada

![softmax](/aprendizagem_nao_supervisionada/algoritmos/Autoencoders/img/softmax.png)

## Esse é basicamente o processo bem simplificado de como funciona o autoencoder

