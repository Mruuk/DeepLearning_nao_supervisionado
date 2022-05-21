### Importação das bibliotecas:

  ```python
from minisom import MiniSom
import pandas as pd
```
### carregamos a base de dados
#### criamos duas variáveis para receber os parâmetros previsores(X) e a classe(y)
```python
base = pd.read_csv('recursos/wines.csv')
X = base.iloc[:,1:14].values
y = base.iloc[:,0].values
```
### normalizamos os valores, para que possam ser trabalhados
```python
# normalização
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)
```
