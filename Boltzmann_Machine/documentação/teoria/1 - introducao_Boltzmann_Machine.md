# Boltzmann Machine

## Teremos duas aplicações práticas

- Sistema de recomendação
- Redução de dimensionalidade
  - Quando se tem uma base de dados com muitos atributos, e precisa escolher qual são os atributos principais.

### Em sua estrutura temos dois neurônios, que vai representar a camada de entrada

#### Ela é a mesma coisa que temos nas redes neurais clássicas

![nós visíveis](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/2neuronios.png)

## Camada dos nós escondidos

### E sua representação é diferente daquela tradicional de redes neurais

![nós escondidos](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/no_escondido.png)

### E nós temos um tipo de ligação diferente

#### Todos ligam-se com todos, incluindo os nós da camada de entrada

- E também há ligação entre a camada de entrada
- É notado que não há uma direção, isso indica que as ligações são para os dois sentidos

![ligacao](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/ligacao.png)

### Olhando para uma rede neural clássica, é notada a diferenção de arquitetura

- Camadas de entradas
- 4 camadas ocultas
- uma camada de saída

## Processo Feedforward

### Para redes neurais clássicas

- A alimentação começa na camada de entrada para a camada de saída
- Faz o cálculo do erro utilizando o backpropagation e retorna para camada de entrada
- Seguindo esse processo, onde a informação vai trafegar em uma só direção, da camada de entrada para de saída

![redeneuralclassic](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/rede_neural_classica.png)

## No cado do Boltzmann, não temos um unico sentido para onde a informação vai percorer

- Onde a própria entrada também se atualiza
- O valor de entrada não vai permanecer fixo, diferentemente das redes neurais clássicas

### Características de sua arquitetura

- Não possui camada de saída
- A premissa é que os nós de entrada também geram dados
- **Descreve o estado do sistema, ajustando os pesos do sistema**(principal motivo do algoritmo)
- Depois do treinamento, pode monitorar o sistema

## Como é uma técnica de redes não supervisionadas

### Que é aquele tipo de aprendizagem que não tem a figura de um supervisor, não existe a **classe**, para um determinado registro, nâo quer fazer nenhum tipo de previsão

#### A boltzmann machine, vai representar o estado de um determinado sistema

- Isso que determina que o bolztman machine se encaixa num aprendizagem não supervisionado

### Com isso vai ser possível fazer de certa forma algum tipo de previsão, ou detectar outliers

## Entender melhor seu funcionamento

### Temos esse avião, onde possui um sistemas, como pode ser visto na imagem abaixo:

![aviao](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/aviao.png)

### Supondo que tenhamos todos os dados gerados por cada um desses 250 componentes

#### Jogando esses dados para uma boltzmann machine, vai ser possível fazer a aprendizagem, do comportamento padrão do sistema

#### Caso tenha algum tipo de falha, será possível detectar que existe uma anomalia no sistema, porque ele aprendeu qual é o comportamento normal desse avião ou sistema, e se tiver alguma coisa fora do padrão, será possível detectar

##### Então é muito utilizado para detectar outleirs

##### Podendo aplicar para vários tipos de indústrias, que tenhamos um comportamento padrão, que a boltzmann machine vai aprender, e ele vai conseguir verificar, por exemplo, se há algum sensor estragado, ou se algum componente ta superaquecendo, então ele vai conseguir detectar esse tipo de padrão

### E claro que essas representações, são mais no geral, por possuir bastante complexidade fazer uma implementação, devido ao fato de que, conforme vai ter maior número de nós, o número de conexão também vai aumentar exponencialmente

#### Supondo que temos uma base de dados com mil atributos, principalmente para imagem, onde posuimos os valores dos pixeis. O tamanho dessa máquina vai ficar muito grande, e vai ficar inviável computacionalmente, para se programar uma máquina dessa.

#### E é por esse motivo que temos as Restricted Boltzmann Machines

## Restricted Boltzmann Machines (RBM)

- Temos os nós visíveis e os nós escondidos
- E a representação já está mais linear
- E temos as ligações, que é bem semelhante ao que temos nas redes neurais tradicionais
- A diferença da RBM para a boltzmann machine clássica é
  - na clássica temos uma ligação todos com todos
  - na RBM, não temos a ligação da entrada com a entrada, e nem entre os neuronios da camada oculta
- Por isso chamamos de restricted, pois está restringindo o tamanho dessa máquina, assim, conseguindo computacionalmente executar ela com  mais eficiencia

![RBM](/aprendizagem_nao_supervisionada/algoritmos/Boltzmann_Machine/documenta%C3%A7%C3%A3o/img/RBM.png)
