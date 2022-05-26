# Redução de dimensionalidade de imagens

## iremos fazer uma redução de dimensionalidade, onde ao invés de usarmos todos os pixels para fazer as supervisões, vamos utilizar uma rbm para reduzirmos a dimensionalidade

- Como exemplo: ao invés de usar uma dimensão 28x28, podemos usar uma dimensão 5x5, e com isso conseguiremos reduzir a quantidade de informações, que vai facilitar o tempo de processamento
- E por sua fez iremos verificar, se essa abortagem trará melhores resultados

## Realizamos algumas importações

- importamos a base mnist atraves do sklearn, com o datasets, da mesma forma que já foi realizada com o keras
- importamos o metrics, veremos o resultado da classificação, usando uma abordagem e outra
- importamos BernoulliRBM, é a implementação das restricted boltzmann machines no sklearn, Bernoulli é um matemático, onde tem a sua equação de bernoulli, que é usada nesse tipo de algoritmo
- importamos GaussianNB, é um algoritmo utilizado para fazer estimativas de probabilidades, é um dos algoritmos mais basicos que existe sobre machine learning
- importamos Pipeline, vamos utilizar essa classe, que vai nos possibilitar executar varios processos em conjunto

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
```

## Carregamos a base, o atributo data é o que nos interessa e temos o atributo target

### previsores instanciado e já transformando em float

### classe recebe target do dataset/base

```python
base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target
```

### Realizamos a normalização dos valores

### e dividimos previsores e classe para o treinamento e test

```python
normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size= 0.2, random_state = 0)
```
