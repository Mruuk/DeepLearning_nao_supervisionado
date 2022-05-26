# Redução de dimensionalidade com RBM

## rbm -> o random state é apenas para sempre gerar mesmo resultado

- Primeiro parametro, n_iter
  - Representa o número de epócas
- Segundo parametros, n_components
  - Representa o número de neurônios da camada escondida

## naive bayes

## O classificador recebe o Pipeline, onde nos permite rodar mais processos, neste caso, iremos rodar o rbm, para reduzir o dimensionamento e logo em seguida iremos rodar o naive bayes

- Fit para treinarmos

```python
rbm = BernoulliRBM(random_state= 0)
rbm.n_iter = 25
rbm.n_components = 50
naive_rbm = GaussianNB()
classificador_rbm = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])
classificador_rbm.fit(previsores_treinamento, classe_treinamento)
```

## vamos plotar as imagens com o subplot, onde teremos varias imagens juntas

- definimos o tamanho das imagens, 20x20 com o figsize
- rbm.components, traz varios valores dos neurônios, definidos no processo anterior
  - iremos percorrer esses neurônios e imprimir as imagens
- subplot tamanho da imagem, 10x10 e indicamos o índice das imagens com o contador + 1
- imshow faremos um reshape de 8x8, redimensionar para o tamanho original, onde era 64(8x8) e transformamos com rbm em 50 e mostraremos as imagens em tons de cinza
- ticks  serve para retirar o caption, dos eixos

```python
plt.figure(figsize = (20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()
```

![subplots](/deepLearning/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/subplot.png)
