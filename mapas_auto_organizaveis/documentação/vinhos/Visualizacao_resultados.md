# Como colocar nas caixas do gráfico cada um dos registros

```python
# visualizar resultados
from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T) # MID - mean inter neuron distance
colorbar()
```

### som.winner, vai dizer qual é o neurônio ganhador de cada registro, BMU
### markers, lista com os marcadores 'o'=circulo, 's'= quadrado, 'D'=losango
### color, lista com os marcadores de cor, 'r'=vermelho, 'g'=verde, 'b'=azul 

```python
w = som.winner(X[0])  
markers = ['o','s', 'D']
color = ['r', 'g', 'b']
```

### transformação dos valores da classe(y) de (1,2,3) em (0,1,2)
- pois se mantivermos a classe começando no 1, os marcadores não pegaram o parâmetro 'o' e 'r'.
- se não ele não vai associar corretamente

```python

y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2
```

### for para percorrer todos os registros e fazer as marcações, pegando o BMU de cada registro/linha
- mostrando o que ocorre

```python
for i, x in enumerate(X):
    #print(i)
    #print(x)
```

### for fazendo o processamento correto
- passaremos também o plot com algums parametros
- passamos w na posição 0, a cordenada x. O '+ 0.5' vai colocar no meio do quadrado do grafico
- passamos w na posição 1, a cordenada y. e o '+ 0.5' para centralizar
- passamos os marcadores com y[i] = a classe com a variavel contadora do for, para percorrer todos os registros, do 0 ao 177
- passamos markerfacecolor, que é a cor da fonte, não vamos passar nenhum parâmetro
- passamos markersize, o tamanho do marcador
- passamos markeredgecolor, parâmetro para preencher a cor dos simbolos, e novamente y[i], para percorrer todo o for
- passamos markeredgewidth, para configurar a borda

```python
for i, x in enumerate(X):
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)
```
![plot][ref]
### visualização dos registros e seus agrupamentos
[ref]: /aprendizagem_nao_supervisionada/algoritmos/mapas_auto_organizaveis/Agrupamento_vinhos/img/visu_with_markers.png "visualização"