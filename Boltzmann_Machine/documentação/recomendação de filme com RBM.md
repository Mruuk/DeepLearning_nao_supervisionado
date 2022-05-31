# recomendação de filmes

## Criamos dois usuários com os filmes que asistiram, importante ser uma matriz

```python
# nparray em formato de matriz
usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])
```

## O run visible é uma função que mostrará qual dos neurônios foi ativado

- se for ativado o neurônio com índice 0, ficará (1,0)
- se for ativado o neurônio com índice 1, ficará (0,1)
- Lembrando que o neurônio com índice 0, representa filmes de comédia e o com índice 1, representa filmes de terror

```python
rbm.run_visible(usuario1)
rbm.run_visible(usuario2)
```

### Output 1

#### Para o usuário 1, o neurônio de índice 0 não foi ativado, e temos o segundo neurônio ativado, que representa as características de filme de terror

#### Já para o usuário 2, o primeiro neurônio foi ativado e o segundo não, representando assim, as características de filmes de comédia

```python
rbm.run_visible(usuario1)
Out[6]: array([[0., 1.]])

rbm.run_visible(usuario2)
Out[7]: array([[1., 0.]])
```

### Recomendando para cada usuário, com base no vetor

```python
filmes = ["A Bruxa", ' A Invocação do mal', 'O Chamado',
          'Se beber não case', 'Gente grande', 'American pie']
```

#### np.array([0,1]), é pro resultado de ativação para o primeiro registro, os de filme de terror

- podemos também criar uma variavel para pegar o resultado que vem da run_visible, e instanciar na recomendação

### no usuario1 temos, os seguintes filmes asistidos, [1,1,0,1,0,0], e nossa recomendação temos o seguinte, [1,1,1,0,0,0]. Perceba que no terceiro filme, foi de 0 para 1, isso é uma recomendação. Nosso usuário, não havia assitido ainda o terceiro filme e o algoritmo percebeu quele tem uma tendência para filmes de terror, e recomendou para ele

```python
camada_escondida= np.array([[0,1]])
recomendacao = rbm.run_hidden(camada_escondida)
```

```python
for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
    if usuario1[0,i] == 0 and recomendacao[0,i] == 1:
        print('usuario 1: ', filmes[i])
```

### Output 2

```python
Out[8]: usuario 1:  O Chamado
```

```python
camada_escondida= np.array([[1,0]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario2[0])):
    #print(usuario1[0,i])
    if usuario2[0,i] == 0 and recomendacao[0,i] == 1:
        print('usuario 2: ', filmes[i])
```

### Output 3

```python
Out[9]: usuario 2:  O Chamado
Out[10]: usuario 2:  American pie
```
