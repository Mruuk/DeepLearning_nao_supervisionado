# Alguns tipos de sistemas de recomendação

- Recuperação direta da informação
  - baseados em Palavras-chave
  - "Os mais vendidos", "Os mais clicados"
  - geralmente se ver em sites de comercio eletronicos, de certa forma não existe muita inteligência nesses tipos de sistema, pois basta uma consulta sql na base de dados, e ja consegue essas informações dos mais vendidos e mais clicados

- Filtragem por conteúdo
  - Conteúdo dos itens
  - Comparação da descrição dos itens

- Filtragem colaborativa
  - uma das tecnicas mais utilizadas hoje em dia, para sistemas de recomendação

## Filtragem Colaborativa - User-base filtering

- Perguntar para amigos
- Prever o grau de interesse sobre um item baseado nas avaliações feitas por esse cliente e as avaliações feitas por outros clientes com perfil semelhante
- Tarefa cooperativa entre um ou mais indivíduos
- Troca de experiências entre perfis comuns

### passos

1. Calcular a similaridade entre os usúarios
2. Calcular as recomendações para os filmes não assistidos
