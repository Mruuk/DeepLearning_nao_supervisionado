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

