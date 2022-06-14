# Redes Adversariais generativas

## um dos tópicps mais avançados que existem na área de deep learning, estão aparecendo muitas aplicações práticas nesse sentidos

- Rede neural que pode criar por ela mesma
  - com forme vimos os outros tipos de redes neurais, ele não tem uma imaginação digamos assim, simplismente posso utilizar para fazer tarefa de classificação, por exemplo, classificar se um cliente vai pagar ou não vai pagar o empréstimo ou  fazzer previsão do preço de um veículo, baseado nas características, fazer classificação de imagens. Então pricipalmente quando trabalhamos com redes neurais convolucionais que passamos imagens, ela aprende a classificar se por exemplo, é um gato ou um cachorro. Esse tipo de rede neural vai gerar imagens, então por exemplo, se passarmos imagens de gatos, ela não vai aprender como uma rede convolucional, a classificar se é gato ou cachorro, ela vai aprender a criar novos gatos baseados nos gatos que recebeu como treinamento
- Aprendem sobre os objetos do mundo e criam outras versões desses objetos que nunca existiram(como se fosse uma imaginação)
- Podem criar imagens a partir de textos
  - ou também criar textos, por exemplo, notícias, ou então baseados nos textos de um determinado poéta, ele pode gerar poesias com as mesmas características desse determinado poeta

### Aqui temos alguns exemplos, onde veremos como que esse tipo de rede neural pode ser utilizado

- Aumento de resolução
  - considere uma imagem bem pequena, onde quer aumentar a resolução, por exemplo, se arrasta a imagem e tenta aumentar o tamanho manualmente, ela vai ficar toda distorcida, então podemos utilizar essa técnica para aumentar a resolução
  - cada uma das imagens representam uma abordagem, e são um comparativo para o SRGAN, onde é a propósta dos autores desse artigo
  - fonte: [artigo](https://arxiv.org/pdf/1609.04802.pdf)
![resolucao](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/resolucao.png)

- Desenho automático
  - implica em desenharmos e a rede neural irar gerar desnhos como base no que foi desenhado
  - como indicado na imagem
  - fonte: [github](https://github.com/junyanz/iGAN)
![draw](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/draw.png)

- Texto para imagem
  - converte texto para imagem
  - com base no texto descritivo é gerado uma imagem de acordo
  - fonte: [artigo](https://arxiv.org/pdf/1605.05396.pdf)
![txt2img](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/txt2img.png)

- Tradução de imagem para imagem
![img2img](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/trad_img2img.png)

- Geração de imagens(google Deep Mind)
  - projeto bastante famoso do google
  - geração automática de imagens
![deepmind](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/deepmind.png)

- Criação de ambientes
  - em uma construção de uma nova casa, se planeja os ambientes, geralmente hoje em dia, temos pessoas que fazer os desenhos dos moveis e ambientess da casa, podemos utilizar esse tipo de rede neural, para gerar varias opções
  - fonte: [artigo](https://arxiv.org/abs/1511.06434)
![criação de ambiente](/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/ambiente.png)
