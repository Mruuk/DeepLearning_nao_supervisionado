# comparando resultados rbm X sem rbm

## .predict para realizar o teste

## metrics para verificar a accuracy do teste

```python
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)
```

## Tivemos uma precisão de 0.8888888888888888, com o RBM + naive bayes

## Agora vamos verificar apenas o naive bayes, sem o rbm, qual será a accuracy

```python
naive_simples = GaussianNB()
naive_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_naive = naive_simples.predict(previsores_teste)
previsao_naive = metrics.accuracy_score(previsoes_naive, classe_teste)
```

## Tivemos uma precisão de 0.8111111111111111, apenas com o naive bayes, notadamento a redução de dimensionamento promove uma precisão melhor

### Em geral teremos resultados melhores
