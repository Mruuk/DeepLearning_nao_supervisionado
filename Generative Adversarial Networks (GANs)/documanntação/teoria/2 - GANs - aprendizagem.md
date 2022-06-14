# Aprendizgem de uma GAN

## principal exemplo para esse tipo de rede neural é que ele vai aprender a criar imagens automaticamente

## Esse tipo de técnica possui basicamente dois componentes

1. Gerador
   - vai gerar imagens
2. Discriminador
   - vai acessar as imagens criadas e informa se elas são parecidas com as originais

- Começam do zero, tanto o gerador quanto o discriminador, e vão aprender sozinhos, o gerador precisa aprender a gerar imagens e o discriminador precisa aprender a avaliar as imagens

### Gerador

- gerador vai receber números aleatórios, pennse em um vetor com 100, 200, 300, números, e por meio desses números ele vai gerar imagens, em sua primeira rodada ele vai gerar uma imagem onde não não vai se asemelhar com um cachorro, por exemplo, e com base nas melhorias dos pesos ele trará resultados mais próximos de um cachorro

### Discriminador

- ele vai receber como parâmetro imagens de cachorros, ele  vai construir uma rede neural, que vai aprender como reconhecer cachorro e  também vai receber outras imagens que não são de cachorros, e com isso ele vai conseguir diferenciar, o que é um cachorro e o que é umnão cachorro. Perceba que é um problema de classificação binária, temos duas classes, a classe cachorro e a classe não cachorro, semelhante a ideia de uma rede neural convolucional, onde que fazer a identificação se uma imagem é de um cachorro ou se é de um gato, no caso desse nosso exemplo, nós continuamos com duas classe, com a simples diferença, onde temos a classe cachorro e a outra será não cachorro

## passo-a-passo de duas rodade de uma aprendizagem de uma rede neural adversarial generativa

- tanto o gerador, quanto o discriminandor, seram redes neurais, onde pode colocar redes neurais densas e convolucionais

![passo-a-passo](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/passo-a-passo.png)

1. no primeiro processo, o gerador vai receber dados ou números aleatórios, e com base nesses números ele vai gerar algumas imagens
   - nessa oemeira rodada, percebe que ele não sabe o que são cachorros, então vai gerar simplesmente imagens ou números aleatórios
   - essas imagens são geradas atraves de pixels, que vao de 0 a 255, e basicamente uma imagem é uma matriz de números, e como estamos passando números, a rede neural vai com base neles, construir determinadas imagens
![primeira-rodada](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/primeira-rodada.png)

2. passamos essas imagens geradas, para o discriminador, que por sua vez vai aprender que as imagens são imagens de não cachorro
![primeira1-rodada](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/primeira1-rodada.png)

3. vamos precisar ter uma base de dados com imagens de cachorros e será passada também como entrada para essa rede neural
   - então temos duas classes, um problema de classificação binária, em que temos imagens de cachorros e de não cachorros
   - perceba que temos apenas um neurônio na camada de saída, por ser um problema de classificação binária, e vamos utilizar a função de ativação sigmoid, uma probabilidade entre 0 e 1
   - e nessa etapa o discriminador não sabe o que são imagens de cachorros, ele não foi treinado ainda e por isso ele vai retornar a probabilidade das imagens serem de cachorros
   - saída esperada de imagens não cachorro é de 0
   - saída esperada de imagens de cachorro é de 1
![primeira2-rodada](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/primeira2-rodada.png)

4. recebendo as entradas o que ele vai fazer é o processo de back-propagation
   - ele vai calcular o erro e vai voltar para camada de entrada e vai atualizar os pesos
   - nessa etapa, ele vai simplismente utilizar uma rede neural tradicional, que vai aprender a diferenciar, o que é cachorro e o que não é cachorro
   - depois do treinamento feito do discriminador, lembrando que ele vai avaliar se as imagens geradas são ou não são cachorros
   ![back-propagation](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/back-propagation.png)

5. Nessa etapa agora, não precisamos mais das imagens dos cachorros, pois vamos considerar que a rede neural discriminadora já foi treinada, e vamos passar as imagens que sabemos que não são de cachorros para a rede neural fazer a classificação
   - entao agora ele vai comparar essas imagens, ele vai pegar esses resultados e comparar com 1 e vai fazer o calculo do erro
   ![comparar](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/comparar.png)

6. Depois de classificadas ele vai pegar esses resultados e vai mandar para primeira rede neural, mais ou menos com se fosse uma conversa entre essas duas redes neurais, então ela vai passar essas imagens para a rede neural discriminadora e ela por sua faz, vai dizer que as imagens passadas então com um erro muito alto e então precisa melhorar as imagens, ela pode indicar do que um cachorro precisa de características, para que possa ser classificado como tal. Feito esse feedback para o gerador, ele vai iniciar o processo novamente
   - e perceba que na segunda rodada ele já vai gerar imagens mais parecidas com de cachorros

 ![segunda-rodada](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/segunda-rodada.png)
7. Então ele vai fazer o mesmo processo, vai pegar as imagens de cachorros originais onde sabemos que são realmente cachorros e a saída do gerador, continuamos com imagens de não cachorros, e passando novamente essas imagens ele vai fazer o treinamentos do discriminador novamente
    - e ainda temos a saída esperada do gerador igual a 0, pois, mesmo as imagens tendo tido melhorias com relação as anteriores, ainda não são imagens de cachorros
    - feito isso ele vai fazer a plicação do algoritmo back-propagation, onde vai rodar por várias épocas, para que essa rede neural do discriminador fique especializada em classificar o que é cachorro e não cachorro
    - será feita a comparação com o valor 1 dessas saídas geradas do discriminador
![segunda-rodada-treinamento](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/treina.png)
8. Agora vai pegar os resultados e novamente vai dar o feedback para a rede neural geradora
    - ele vai pegar os dados do erro e vai fazer todo o processo de treinamento da rede neural geradora e vai gerar novas imagens
![feedback](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/feedback.png)

## E assim vai se repetindo esses processos, até atingir o resultado esperado
