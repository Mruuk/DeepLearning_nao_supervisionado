# recomendações

## Partindo do ponto, onde ao treinar a boltzmann machine, foi encontrado oum nó especialista em filme de terror, e um outro nó especialista em filme de comédia

### Temos o exemplo de apenas um registro, supondo que essa pessoa

- gostou de 'A Bruxa'
- não assistiu 'Invoncação do mal'
- gostou de 'O Chamado'
- não gostou de ' Se Beber não case'
- não gostou de 'Gente grande'
- não assistiu 'American pie'

![recomendacao](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/recomendacao.png)

#### Vamos agora, pegar cada um desses meu neurônio da camada oculta, e comparar, com cada uma das entradas

![terror](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/terror.png)

#### Note que essa pessoa gostou desses dois filmes, então é como se ele acendesse esse neurônio de terror, para esse registro específico

#### Agora vamos verificar o neurônio de comédia, note que os dois filmes assistidos de comédia, foi marcado como 'não gostou', e com base nessa comparação, foi determinado que o neurônio de comédia, não acendeu

##### Essa é a primeira parte da recomendação, onde teremos os neurônios verde e vermelho, onde vermelho é para aqueles que 'não gostou' e o verde para aqueles que 'gostou'

#### E na segunda parte, vamos pegar os filmes cujo, não foram assistidos e vou descobrir se ele vai ser recomendado ou não

1. primeiro passo, verificar qual o tipo do filme
2. filme de terror
3. e temos o neurônio de terror
4. então ele estará ligado apenas ao neurônio de terro da camada oculta
5. verifica se o neurônio ta verde ou vermelho
6. como está verde, foi aceso, com base nas preferencias do usuário, iremos recomenda-lo este filme, e colocaremos o valor 1 no neurônio de entrada

![terror recomendado](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/recomendacao2.png)

#### Agora vamos verificar se recomendamos 'American pie' ou não

1. verifica qual é o neurônios que é filme de comédia
2. feita a ligação, é verificada a preferencia do usuário
3. colocamos o valor 0 em 'American pie', devido a preferencia do usuário, note que o neurônio está vermelho, e portando não recomendaremos filmes de comédia para esse registro/usuário

![comedia recomendado](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/recomendacao3.png)

## Estruturas de redes neurais

![estrutura de redes neurais](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/estruturas%20neurais.png)
