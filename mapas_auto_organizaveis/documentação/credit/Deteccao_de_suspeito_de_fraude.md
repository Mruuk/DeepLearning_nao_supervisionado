# Análise o mapa auto organizável, buscando clientes outliers

### Gráfico de agrupamento dos clientes

![plot](/aprendizagem_nao_supervisionada/algoritmos/mapas_auto_organizaveis/fraudes/img/plot.png)

#### Mapeamento é um dicionário, ele tem os valores em cada posições

#### Cada linha dele tem os registros ou os clientes que foram agrupados em sua semelhança, eles escolheram a posição como o neurônio vencedor o BMU

#### Trazendo cada um dos registros associados a o BMU

#### Vamos concatenar os neurônios(vetores)

- Olhando para o gráfico observamos os MID onde também são BMU, entao temos uma grande chance de termos outliers, ou fraudadores, por estar muito fora do padrão.
- Escolhemos os dois vetores e realizamos a concatenação
- axis = 0, para realizar a concatenação em colunas, colocando um em baixo do outro

#### Necessário inverter a normalização, para que possamos indentificar os clientes

```python
mapeamento = som.win_map(X)
suspeitos = np.concatenate((mapeamento[(4,5)], mapeamento[(6,3)]), axis = 0)
suspeitos = normalizador.inverse_transform(suspeitos)
```

#### Iremos agora verificar a classe de cada um dos registros da variável suspeitos

#### Pois se o valor da base para um suspeito for 0, quer dizer que ele teve crédito aprovado, e se esse for o caso, é preciso fazer algum tipo de avaliação

#### Porém, se um suspeito recebe o valor 1, para crédito reprovado, não vai ajudar muito a empresa, pois não oferece risco para ela

#### Rodamos um for para percorrer a base inteira e um outro para os suspeitos

- Com isso precisamos de im if para pegarmos os ids dos clientes que estão como suspeitos
- Necessário um arredondamento e transformação no tipo inteiro na variável suspeitos, pois pode ocorrer dos valores não serem exatos, impedindo a comparação e consequentimente a associação das base e suspeitos

```python
classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 4])
```

#### Convertemos a classe em array, para que possamos manipular mais facilmente

#### Vamos novamente concatenar suspeitos e a classe, a classe são os valores de aprovado ou reprovado empréstimo/crédito, que estão associados aos suspeitos

#### Agora apenas ordenamos com base na coluna de aprovação e reprovação

```python
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]
```

#### Por fim, salvando o resultado para encaminhar para o departamento de fraudes

```python
df = pd.DataFrame(suspeitos_final)
df.to_csv('suspeitos_de_fraude.csv', index=False)
```
