# Construindo o mapa auto organizável - Vinhos

- Primeiro parâmetro é o valor de x, que são quantas **linhas** que o mapa vai ter
  - recebeu valor de 8
- Segundo parâmetro, é o valor de y, que são as **colunas** que o mapa vai ter
  - recebeu valor de 8
- Foi definido o valor de 8 x 8 conforme a [teoria](#apoio)
- temos 178 registros na base
- Terceiro parâmetro, é o input_len, ou seja, quantas entradas teremos, que é o número de **atributos**
  - neste caso, o valor é de 13
- Quarto parâmetro, é sigma, podemos colocar igual a 1, que é o valor default
  - Ele equivale ao raio, vale ao alcance dos neurônios ou a quantos neurônios, baseado no BMU
        - o neurônio vencedor ou Best Matching Unit, ele vai utiliza os BMU, para traçar o raio, para fazer a atualização de todos os neurônios que estão em volta
- Quinto parâmetro, é o learning_rate, que é a taxa de aprendizagem
  - Quando fazemos a atualização dos pesos, e neste tipo de rede neural consiste em pegar o BMU e atualizar o peso dele, para deixa-lo mais próximo possível da entrada/registro da base de dados
- Sexto parâmetro, é o random_seed, esse parâmetro é resposável por manter sempre o mesmo resultado, toda vez que seja executado o código
  - A inicialização dos pesos sempre terá o mesmo valor

```python
    som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2) 
```

### Inicializando pesos

```python
    som.random_weights_init(X)
```

### Treinamento

- num_iteration, é o número de repetições ou de **épocas**
  - Em geral, colocar o valor de 100 é suficiente para a maioria dos casos
  - A cada época ele vai aproximar os pontos dos dados originais(entradas), reduzindo o raio até acabar o número de épocas

```python
    som.train_random(data = X, num_iteration = 100)
```

### avaliando os dados

- _weights, mostrando que posuimos uma matriz 8 x 8
- _activation_map, valores do mapa auto organizáveil
- activation_response(X), visualizamos quantas vezes cada um dos neurônios foi selecionado como o BMU

```python
    som._weights
    som._activation_map
    q = som.activation_response(X)
```
 
<h1  align="left"><img src="/home/lisboa/learn/deepLearning/aprendizagem_nao_supervisionada/Teoria - Mapas auto organizáveis/apoio_/activation_response.png"/></h1> 

## Visualizar os resultados

- Retorna uma matriz com os valores de distância e o T é de transposta, precisamos disso para usarmos a função pcolor
  - MID: vamos tirar a média da distância enclidiana, a média da disntância do neurônios, ele vai computar a distancia entre um determinado neurônio e a distância dos neurônios a sua volta
  - O qual parecido o neurônio é dos seus vizinhos

```python
    from pylab import pcolor, colorbar
    pcolor(som.distance_map().T) # MID - mean inter neuron distance
    colorbar()
```

## MID - mean inter neuron distance

- Na escala 1 representa o mais diferente possível dos vizinhos e 0, o mais semelhante

<h1  align="left"><img src="/home/lisboa/learn/deepLearning/aprendizagem_nao_supervisionada/algoritmos/mapas_auto_organizaveis/Agrupamento_vinhos/img/MID.png"/></h1>

## Apoio: 

### Tamanho do SOM(Self-organizing map):

- $tamanho = 5\sqrt{N}$
- Base com 178 registros
- $tamanho = 5\sqrt{178}$
- $tamanho = 5 \cdot 13,11$
- tamanho = 65,65 células/neurônio
- Sendo uma matriz $8\times8$
- Consultar autor Vesanto, materia de mapas auto organizáveis