# Treinamento e comparação da assertividade

1. primeiro criamos um rede neural classificadora para os dados sem redução de dimensionalidade
    - lembrando que o seu input deve ser o número de registros, no caso de pixels, que no caso dessa base de daos, temos 784
    - sua camada oculta deve ser 397, seguindo aquele cálculo
    $$(camada\: de\: entrada+camada\: de\: saida)/2
    \quad \rightarrow \quad
    (784+10)/2$$
    - camada de saída com 10 neurônios e função de ativação softmax, pois temos um caso com varias saidas

2. segundo criamos uma rede neural classificadora para os dados com redução de dimensionalidade
    - nesse caso nossa camada de entrada deve ser 32, pois estamos pegando a base de dados com a redução de dimensionalidade
    - camada oculta recebe 21, seguindo o mesmo princípio matemático da anterior

```python
# sem redução de dimensionalidade
c1 = Sequential()
c1.add(Dense(units = 397, activation = 'relu', input_dim = 784))
c1.add(Dense(units = 397, activation = 'relu'))
c1.add(Dense(units = 10, activation = 'softmax'))
c1.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',
           metrics = ['accuracy'])
c1.fit(previsores_treinamento, classe_dummy_treinamento, batch_size = 256,
       epochs = 100, validation_data = (previsores_teste, classe_dummy_teste))

# com redução de dimensionalidade
c2 = Sequential()
c2.add(Dense(units = 21, activation = 'relu', input_dim = 32))
c2.add(Dense(units = 21, activation = 'relu'))
c2.add(Dense(units = 10, activation = 'softmax'))
c2.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',
           metrics = ['accuracy'])
c2.fit(previsores_treinamento_codificados, classe_dummy_treinamento, batch_size = 256,
       epochs = 100, validation_data = (previsores_teste_codificados, classe_dummy_teste))
```

## Output non-reduction classifier

```python
Epoch 100/100
235/235 [==============================] - 13s 54ms/step - loss: 0.0158 - accuracy: 0.9952 - val_loss: 0.0721 - val_accuracy: 0.9804
```

## Output reduction classifier

```python
Epoch 100/100
235/235 [==============================] - 2s 7ms/step - loss: 0.1613 - accuracy: 0.9516 -   val_loss: 0.1720 - val_accuracy: 0.9482
```

### Aqui já notamos de cara que, com a redução, por ter menos valores, o tempo de execução é menor, porém tivemos uma penalidade em nossa assertividade de 3%. O que devemos avaliar nesse caso, é o que deve ser melhor para o sistemas, se demorar mais para fazer o treinamento e teer uma assertividade mais alto, ou demorar menos mas penalizando a assertividade

### Então, se tivermos um sistema que precise ficar treinando, toda hora ta chegando novos dados, e pelo menos uma vez por dia precisa fazer o treinamento, então se utilizar a primeira abordagem, sem a redução de dimensionalidade, ele vai demorar muito, podendo assim ser inviável, porém se utilizar a segunda abordagem, onde temos a redução de dimensionalidade, talvez não tenha um percentual tão grande de acertos, porém teremos uma velocidade no processamento.  Então vale a pena, discutir se a perda de 3% é realmente significativa, porque, dependendo do sistema trabalhado, não há uma necessidade de atingir o 98%, com 95% já consegue ter resultados interessantes
