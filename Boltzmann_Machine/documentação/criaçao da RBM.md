# Criação da restricted boltzmann machine

## Primeiramente importamos a RBM, onde é a implementação da boltzmann machine, já pronta

## Importamos o numpy

```python
from rbm import RBM
import numpy as np
```

## Criamos a nossa restricted boltzmann machine

- passamos a classe contrutora do rbm.py, onde recebe dois parametros
  - num_visible, camada de entrada ou nós visíveis
  - num_hedden, camada oculta, ou nós ocultos

```python
rbm = RBM(num_visible = 6, num_hidden = 2)
```

## Criamos a nossa base, como em nosso exemplo da teoria

## 1 = assisiu e gostou, 0 = não gostou e/ou não asistiu

### Nessa biblioteca não temos o conceito de não ter assistido aos filmes, onde em nossos exemplos da teoria, não recebia valor

## As três priemiras colunas representam filmes de terror e as três últimoas representam filmes de comédia

### temos em cada uma das linha os usuários que assitiram aos filmes, se totalizando 6, como nos exemplos da teoria

```python
base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])
```

## Aplicamos o trainamento, passando a base, e o número máximo de épocas

- fará o gibbs sampling, e durando o treinamento tentará encontrar os neurônios que são especialistas em encontrar determinadas caracteristecas do filme
- O valor 5000, é um valor que foi indicado pelo autor da codificação do rbm.py, onde acima disso, nessa implementação em específico, não trará melhores resultados

```python
rbm.train(base, max_epochs = 5000)
```

## para visualizar os pesos

```python
rbm.weights
```

### Output

#### A primeira coluna indica a unidade de bias, e a primeira linha indica os valores para unidade de bias

#### Cada linha representa um filme, e as duas últimas colunas representam os dois neurônios ocultos. Note que os valores estão separados entre positivos e negativos, onde os positivos, indicam o qual especialista ficou o neurônio, note que para os três primeiros filmes, que são de terror, o nosso segundo neurônio se especializou mais em identifica-los, e o nosso primeiro neurônio se especializou mais em achar filmes de comédia

#### para entender melhor, para cada linha que representa um filme, vamos colocar os filmes do exemplo da teoria

- A Bruxa
- Invocação do mal
- O Chamado
- Se Beber não case
- Gente grande
- American pie

#### OBS: provavelmente foi dado valor positivo para "O Chamado", pois na base de dados, todos que assistiram de comédia, tbm assistiram "O Chamado"

```python
Out[6]: 
array([[ 2.4763908 ,  2.33266285,  0.38099372],
       [-1.01421952, -3.28313784,  7.9407139 ],
       [ 0.2143899 , -7.22240774,  3.31598125],
       [ 4.94562669,  3.90103623,  1.87966662],
       [ 0.94992981,  3.62920907, -8.24088492],
       [-1.31346401,  0.60844544, -4.55303516],
       [ 0.95519188,  3.62516475, -8.2434627 ]])
```

## Foi possivel detectar esses padrões, onde nosso primeiro neurônio é especialista em filmes de comédia e nosso segundo neurônio é especialista em filmes de terror

neurônio(comédia) | neurônio (terror)
------- | -------
-3.28313784 | 7.9407139
-7.22240774 | 3.31598125
3.90103623 | 1.87966662
3.62920907 | -8.24088492
0.60844544 | -4.55303516
3.62516475 | -8.2434627

### Claro que o nosso algoritmo, não tem ideia do que é um filme de terror ou de comédia, até porque, não foi passado nenhum padrão nem nome de filme ou gênero, ele ta descobrindo altomáticamente, é como se fosse um algoritmo de agrupamento, ele acaba agrupando os filmes de comédia e de terror, baseado somente nas notas que os usuários deram

### Com base nas avaliações dos usuários, se consegue fazer esse agrupamento, e com base nesse agrupamento, conseguimos fazer a recomendação dos filmes
