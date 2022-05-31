# construção do mapa

## com o objetivo de detectar os outliers,  ou aqueles clientes que tenha alguma chance de cometer algum tipo de fraude

### O id do cliente nós vamos usar pois vamos recuperar esses dados depois

#### usamos a fórmula para descobrir as dimenções, e os valores de x e y para o minisom

#### Tamanho do SOM(Self-organizing map)

- $tamanho = 5\sqrt{N}$
- $tamanho = 5\sqrt{1997}$
- $tamanho = 5 \times 44,68$
- $tamanho = 223,43$
- $15 \times 15 = 225$

#### O input_len são os parâmetros da base

#### Não colocando o learning_rate e o sigma ele terão o valor default

- O learning_rate default = 0.5
- O sigma default = 1.0

#### O random_seed, colocamos para manter sem rotatividade nos valores, sempre que rodar teremos os mesmos valores

```python
som = MiniSom( x = 15, y = 15, input_len= 4, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
        markerfacecolor = 'None', markersize = 10,
        markeredgecolor = color[y[i]], markeredgewidth = 2)
```

![plot](/aprendizagem_nao_supervisionada/algoritmos/mapas_auto_organizaveis/fraudes/img/plot.png)

- quadrados verdes, são aqueles clientes que não tiveram seu crédito aprovado
- circulo vermelho, são aquele que tiveram seu crédito aprovado
- Existem também registros que escolheram o BMU(best matching unit), aqueles neurônios que tem MID(mean inter neuron distance), alto, e se é alto, provavelmente é um outlier, pois foge muito do padrão.
