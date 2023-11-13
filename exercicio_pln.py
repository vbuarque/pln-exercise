import pandas as pd
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

# Carregar os dados
data = pd.read_csv('bd\data.csv')

# Analise do'Sentence' e 'Sentiment'
reviews = data['Sentence'].tolist()
sentiments = data['Sentiment'].tolist()

# Pré-processamento de dados
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess(review):
    # Tokenização e remoção de palavras irrelevantes e lematização
    words = word_tokenize(review)
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return words

processed_reviews = [preprocess(review) for review in reviews]

# Divisão dos dados em treinamento e validação
train_reviews = processed_reviews[:15]
train_sentiments = sentiments[:15]
validation_reviews = processed_reviews[15:60]
validation_sentiments = sentiments[15:60]

# Construção do modelo Word2Vec
w2v_model = Word2Vec(train_reviews, vector_size=100, window=5, min_count=1, workers=4)
w2v_model.train(train_reviews, total_examples=w2v_model.corpus_count, epochs=10)

# Transformação dos dados de treinamento e validação em vetores usando o modelo Word2Vec
train_vectors = np.array([np.mean([w2v_model.wv[word] for word in review if word in w2v_model.wv.key_to_index], axis=0) if np.sum([word in w2v_model.wv.key_to_index for word in review]) > 0 else np.zeros(w2v_model.vector_size) for review in train_reviews])
validation_vectors = np.array([np.mean([w2v_model.wv[word] for word in review if word in w2v_model.wv.key_to_index], axis=0) if np.sum([word in w2v_model.wv.key_to_index for word in review]) > 0 else np.zeros(w2v_model.vector_size) for review in validation_reviews])

# Classificação usando MLP para Word2Vec
mlp = MLPClassifier(max_iter=2000, learning_rate_init=0.001)
mlp.fit(train_vectors, train_sentiments)
w2v_predictions = mlp.predict(validation_vectors)

# Construção do modelo Bag of Words com transformação TFIDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform([' '.join(review) for review in train_reviews])
validation_vectors = vectorizer.transform([' '.join(review) for review in validation_reviews])

# Classificação usando MLP para Bag of Words
mlp = MLPClassifier(max_iter=2000, learning_rate_init=0.001)
mlp.fit(train_vectors, train_sentiments)
bow_predictions = mlp.predict(validation_vectors)

# Comparação dos modelos
w2v_accuracy = accuracy_score(validation_sentiments, w2v_predictions)
bow_accuracy = accuracy_score(validation_sentiments, bow_predictions)

w2v_prob_mean = np.mean(w2v_predictions == validation_sentiments)
bow_prob_mean = np.mean(bow_predictions == validation_sentiments)

# Tabela comparativa
data = {
    'Modelo': ['Word2Vec (W2V)', 'Bag of Words (TFIDF)'],
    'Percentual de Acerto': [w2v_accuracy * 100, bow_accuracy * 100],
    'Probabilidade Média de Acertos': [w2v_prob_mean * 100, bow_prob_mean * 100 ]
}

df = pd.DataFrame(data)
print(df)