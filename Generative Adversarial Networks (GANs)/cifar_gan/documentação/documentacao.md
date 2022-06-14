# Gan base de daddos cifar10

- O gerador e discriminador possuem mais de 10 camadas (`Dense`, `BatchNormalization`, `Regularization`, `UpSampling2D`, `MaxPooling2D`, `AveragePooling2D` e `Convolution2D`), ou seja, une vários conceitos de vários tipos de redes neurais

- Para ter resultados interessantes, o ideal é executar o código por 100 épocas. Cada época leva em torno de 30 minutos para executar em um notebook com processador i7, ou seja, no mínimo dois dias

- Esse cálculo é baseado somente na base cifar10, que possui imagens com dimensões 32 x 32. Para gerar imagens com dimensões maiores, a rede neural poderá levar **meses** para treinar!

![resultado](/aprendizagem_nao_supervisionada/algoritmos/Generative%20Adversarial%20Networks%20(GANs)/img/imgResultado.png)
