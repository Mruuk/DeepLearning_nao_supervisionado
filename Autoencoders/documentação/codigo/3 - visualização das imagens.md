# visualização das imagens

## vamos criar um novo modelo, para capturarmos somente a aprendizagem da rede, que é efetivamente o codificador, e faremos isso utilizando o Model do keras

- vamos criar dimensao_original, recebe um Input, que é uma camada do keras, onde terá seu tamanho de 784, onde é o tamanho original que iremos receber
- camada_encoder, selecionamos a camada que codifica nossas imagens, e ela é nossa primeira camada, já que a segunda é de decoder
- criamos o encoder, que recebe Model, pegando a dimensão original e a camada de encoder
  - são os parametros de input e output do Model
- chamamos um summary, para verificar a estrutura construida

```python
dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
encoder.summary()
```

- criamos imagens codificadas, que vai receber o predict do encoder
  - note que aqui estamos criando apenas a etapa de codificação de uma imagem
  - recebe 784, conforme definimos, e transforma em 32 dimensões
- e criamos imagens decodificadas, que vai receber o predict de autoencoder
  - note que aqui estamos criando apenas a etapa de decodificação de uma imagem

```python
imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)
```

- agora para visualizarmos os resultados
- numero_imagem recebe 10, pois queremos visualizar 10 imagens
- imagem_teste recebe função que randomiza os valores, juntamente com randint, pegando previsores_teste, ou melhor seu tamanho, pois é o shape[0], vale exatamente 10000, e passamos o numero de imagens que queremos, no caso, 10 imagens
  - 10 imagens em um intervalo de 0 à 10000
- criação do grafico
  - plt figure figsize, trás apenas o tamanho das imagens a serem visualizadas
  - fazemos um for pegando essas 10 imagens randomicas
  - vamos fazer 3 gráficos

1. gráfico, imagem original
    - eixo vai receber subplot, são as 10 imagens, e recebe o número de linhas e colunas, e damos um índice para imagem
    - imshow, para mostrarmos as imagens, passamos previsores_teste, o indice_imagem que percorre o for, e um reshape, para 784 dimensões
    - previsores_teste, foi passada pois são as imagens que não sofreram nenhum tipo de alteração
    - retiramos as barras de informação do x e y
2. gráfico, imagem codificada
    - o eixo se repete, porém o índice tem que ser diferente, para não conflitar com os que foram gerados nas imagens originais
    - imshow, recebe imagens_codificadas, método de codificação criado nos passos anteriores, e sua dimensão será de (8,4), multiplicado, temos 32
3. gráfico, imagem decodificada
    - o eixo se repete, tendo apenas que modificar o índice, para não ter conflico com os demais criados
    - imshow, recebe o método de decodificação criado nos passos anteriores e sua dimensão é alterada para 784 dimensões

```python
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    
    # imagem original
    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10,10,i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
    # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
```
