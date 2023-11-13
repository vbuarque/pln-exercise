# Repositório é reservado a resolução da lista 04 de PLN! 

<br id="topo">

# Base de Dados Utilizada:

<img src="https://github.com/bruno-pisciotta281/pln_ex/blob/main/base%20de%20dados.PNG" width="500px;"/>
(https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis/data)

# Exercício 01:

<img src="https://github.com/bruno-pisciotta281/pln_ex/blob/main/exercicio%2001.PNG" width="500px;"/>

## PP.4.1. Construa um modelo do tipo word2vec (W2V) para classificação de revisões de produto que atenda aos seguintes critérios:

### a) O modelo deve operar sobre dados de revisão que tenham sido pré-processados;
<br />
O pré-processamento dos dados é feito na função preprocess(review). Esta função tokeniza as revisões, remove palavras irrelevantes (stop words) e aplica a lematização nas palavras restantes usando o algoritmo RSLPStemmer.


```python
def preprocess(review):
    # Tokenização e remoção de palavras irrelevantes e lematização
    words = word_tokenize(review)
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return words

processed_reviews = [preprocess(review) for review in reviews]
```

### b) O modelo deve ser comparado, em termos de desempenho de classificação, com um modelo clássico do tipo bag of words (BOW) com transformação TFIDF.
<br />
A comparação entre o modelo Word2Vec e o modelo Bag of Words é feita na parte final do código. Ambos os modelos são treinados e suas previsões são comparadas.

```python
# Comparação dos modelos
w2v_accuracy = accuracy_score(validation_sentiments, w2v_predictions)
bow_accuracy = accuracy_score(validation_sentiments, bow_predictions)

w2v_prob_mean = np.mean(w2v_predictions == validation_sentiments)
bow_prob_mean = np.mean(bow_predictions == validation_sentiments)
```

### c) Para a classificação, utilizar no mínimo 15 reviews de treinamento, classificador utilizando Multilayer Perceptron e mais 45 reviews de validação, sendo elas igualmente distribuídas entre revisões positivas, negativas e neutras.
<br/>
A divisão dos dados em conjuntos de treinamento e validação é feita da seguinte forma:

```python
# Divisão dos dados em treinamento e validação
train_reviews = processed_reviews[:15]
train_sentiments = sentiments[:15]
validation_reviews = processed_reviews[15:60]
validation_sentiments = sentiments[15:60]
```

A classificação usando o Multilayer Perceptron (MLP) é feita para ambos os modelos, Word2Vec e Bag of Words.

```python
# Classificação usando MLP para Word2Vec
mlp = MLPClassifier(max_iter=2000, learning_rate_init=0.001)
mlp.fit(train_vectors, train_sentiments)
w2v_predictions = mlp.predict(validation_vectors)

# Classificação usando MLP para Bag of Words
mlp = MLPClassifier(max_iter=2000, learning_rate_init=0.001)
mlp.fit(train_vectors, train_sentiments)
bow_predictions = mlp.predict(validation_vectors)

```

### Crie uma tabela comparativa entre o BOW e W2V que mostre: O percentual de acerto das reviews e a probabilidade média relacionada aos acertos.

A tabela comparativa é criada e impressa no console no final do código.

```python
# Tabela comparativa
data = {
    'Modelo': ['Word2Vec (W2V)', 'Bag of Words (TFIDF)'],
    'Percentual de Acerto': [w2v_accuracy * 100, bow_accuracy * 100],
    'Probabilidade Média de Acertos': [w2v_prob_mean * 100, bow_prob_mean * 100 ]
}

df = pd.DataFrame(data)
print(df)
```

## Resultado Exercício 01:

<img src="https://github.com/bruno-pisciotta281/pln_ex/blob/main/resultadoex01.PNG" width="500px;"/>

## Conclusão ex01:

O modelo Word2Vec teve um desempenho melhor que o modelo Bag of Words, com uma precisão de 51.11% contra 37.78%. Isso sugere que o modelo Word2Vec pode ser mais adequado para essa tarefa específica. No entanto, a precisão ainda é relativamente baixa, o que sugere que pode haver espaço para melhorias no pré-processamento dos dados, na escolha do modelo ou nos parâmetros do mesmo, porém por conta de limitações relacionadas ao hardware este foi o resultado possível até o momento.

# Exercício 02:

<img src="https://github.com/bruno-pisciotta281/pln_ex/blob/main/exercicio%2002.PNG" width="500px;"/>

## Código:

```bash

import pandas as pd
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Carregue um modelo Word2Vec
word2vec_model = api.load('glove-wiki-gigaword-100')

# Função para encontrar sinônimos com base na similaridade do Word2Vec
def find_synonym(word, word2vec_model):
    try:
        synonyms = word2vec_model.most_similar(positive=[word], topn=5)
        return synonyms[0][0]  # Escolha o sinônimo mais similar
    except KeyError:
        return word

# Função para reescrever a frase do usuário
def rewrite_sentence(sentence, word2vec_model):
    words = word_tokenize(sentence)
    rewritten_sentence = []
    
    for word in words:
        # Verifique se a palavra não é uma stop word e possui sinônimos no modelo Word2Vec
        if word not in stopwords.words('english') and word in word2vec_model.key_to_index:
            synonym = find_synonym(word, word2vec_model)
            rewritten_sentence.append(synonym)
        else:
            rewritten_sentence.append(word)
    
    return ' '.join(rewritten_sentence)

# Carregar os dados
data = pd.read_csv('bd\data.csv')

# Pegue uma amostra dos dados
sample_data = data.head(10)

# Reescreva cada revisão na amostra de dados
rewritten_reviews = sample_data['Sentence'].apply(lambda x: rewrite_sentence(x, word2vec_model))

# Adicione as revisões reescritas ao DataFrame
sample_data['Rewritten_Sentence'] = rewritten_reviews

print(sample_data)

```

## Resultado Exercício 02:

<img src="https://github.com/bruno-pisciotta281/pln_ex/blob/main/resultadoex02.PNG" width="500px;"/>

## Conclusão ex02:

A base de dados utilizada para treinamento e substituição das palavras pode afetar o desempenho do sistema.

Se o corpus de treinamento é grande e diversificado, o modelo pode aprender uma ampla gama de sinônimos para várias palavras, melhorando o desempenho. No entanto, se o corpus de treinamento é limitado ou enviesado, o modelo pode não aprender relações significativas entre as palavras, o que pode prejudicar o desempenho.

No caso a a base de dados utilizada para o treinamento é o modelo Word2Vec pré-treinado ‘glove-wiki-gigaword-100’, que foi treinado em um grande corpus de texto da Wikipedia e da Gigaword. A qualidade e a diversidade do corpus de treinamento (Wikipedia e Gigaword) podem ter ajudado o modelo a aprender uma ampla gama de sinônimos para várias palavras, melhorando o desempenho. No entanto, se muitas das palavras na base de dados (Financial Sentiment Analysis - Mostrada anteriormente) não estiverem presentes no vocabulário do modelo Word2Vec, o sistema pode não ter sido capaz de encontrar sinônimos para essas palavras, o que pode ter afetado o desempenho.

Portanto, a escolha da base de dados para treinamento e substituição das palavras é um fator importante que pode afetar o desempenho do sistema, no caso deste exercício, foi a base possível de ser trabalhada, levando em conta as limitações de Hardware e disponiblidade quanto aos requsitos. 


## Como rodar o projeto

1. Clone este repositório para a sua máquina local usando `git clone https://github.com/vbuarque/pln-exercise`.

2. Entre na pasta do projeto `cd pln-exercise`

3. Instale as bibliotecas necessárias usando o comando `pip install -r requirements.txt`.

4. Execute o script Python usando o comando `python ex01.py` ou `python ex02.py`.


<p align="right"><a href="#topo">Voltar ao Topo</p> 

