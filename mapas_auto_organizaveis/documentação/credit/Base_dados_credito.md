# Vamos iniciar ao um exemplo de mapas auto organizaveis

## Onde vamos tentar fazer a detecção de suspeitos de fraudes

### a base de dados possui

- clienteid
- income, quanto que ela possui
- age, sua idade
- loan, quanto que deve
- default, representa inadiplante, se foi ou não aprovado para crédito bancário/empréstimo
  - o aprovou, 1 não aprovou

### importações

```python
import pandas as pd
from minisom import MiniSom 
import numpy as np
```

### Preprocessamento

- Retiramos atributos que são nan
- Temos também, valores negativos no age
  - usamos o a média para substituir esses valores negativos
  - base.age.mean(), para saber a média de idade

```python
base = pd.read_csv('recursos/credit_data.csv')
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

```

### Normalização

```python
from sklearn.preprocessing import MinMaxScaler

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

```
