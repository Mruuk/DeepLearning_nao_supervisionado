# Aprendizagem RBM

## Base é um sistema de recomendação, para entendermos o funcionamento

### Como é o exemplo da netflix, que lhe recomenda/indica um filme

![representacao](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/representacaoRBM.png)

- 6 entradas
- 2 neurônios na camada oculta
- é pouco, colocar apenas 2 neurônios, assim como 6 na camada de entrada, mas estamos com essa configuração apenas para fins didáticos

#### Temos as ligações

![ligacao](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/RBMligacao.png)

- Consideramos que cada neurônio da camada de entrada é um nome de um filme

![basedados](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/basedados.png)

- uma base de dados com 6 usuários
- as colunas estão relacionadas com as entradas da rede neural
- Ana assistiu, A Bruxa e O Chamado e gostou, e assistiu, Se Beber não case e American pie, e ela não gostou, e ela não assistiu Invocação do mal e nem Gente grande

#### Os três primeiros filmes são de terror e os três últimos são de comédia

![terror x comedia](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/terrorxcomedia.png)

#### O que vai acontecer é que a boltzmann machine vai aprender que um dos neurônios indica filmes de terror, enquanto o outro vai indicar filme de comédia

- Isso não é definido por nós, o próprio sistema vai definir, utilizando todos os dados coletados da base.
- É como se ele criasse um neurônio para representar cada categoria de filme
- Esse algoritmo é como uma caixa preta, não é possível determinar qual neurônio é encarregado de classificar um tipo específico de filme
- o algoritmo vai pegar os dados, passado pela base de dados, e ele vai encontrar os padrões desses dados, e vai criar neurônios especialistas em determinadas características
- Aprende alocar os nós escondidos de acordo com as características(cada nó é um padrão)
- Interconectividade entre a nota dos usuários, vai permitir que ele construa nós especialistas em identificar determinadas característica
- Provavelmente existe alguma característica que os filmes possuem que faz as pessoas gostarem(pessoas gostam das características)

![outro exemplo](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/filmes.png)

### Para chegarmos nesse objetivo

#### Utilizamos Constrative divergence(aprendizagem)

- É o algoritmo utilizado na aprendizagem desse tipo de técnica
- o processo consiste em fazer o cálculo dos pesos, vai definir pesos aleatórios
  - A ideia de inicialização de pesos é semelhando ao outros tipos de redes neurais
- iremos calcular os valores, e cada um dos neurônios terá um valor

### O que ele vai fazer é um processo de reconstrução

#### Ele tendo os valores dos neurônios, ele vai reconstruir os registros. Colocando valores para cada um deles, depois disso ele vai pegar essa nova entrada

- observação: peceba que aqui ele ta modificando a entrada, diferentemente dos conceito clássico de rede neural, que a entrada fica fixa

![constrative divergence](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/constrativeDivergence.png)

#### Depois de reconstruir a nova entrada, ele vai pegar os valores reconstruidos e jogar para a camada oculta, onde ele fará um cálculo matemático e vai reconstruir novamente os dados e assim vai seguindo esse fluxo até um determinado número de épocas, ou então, quando  os valores das entradas, for igual as entradas originais. É como uma espécie de encoders, por meio dos valores da camada oculta, ele vai tentar gerar o valor original da camada de entrada

- Como se fosse uma redução de dimensionalidade, reduzindo de 6 neurônios, para 2 e atraves desses 2, ele vai tentar retornar ao valor original dos 6 neurônios

![constrative divergence](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/constrativeDivergence2.png)

### Reconstrução do nó,(gibbs sampling)

- Gibbs sampling, é um conceito la da matemática

![gibbs](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/gibbs.png)
