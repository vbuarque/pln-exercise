import re
import nltk
nltk.download('rslp')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

reviews = ["Eu amo este produto", "Este é o pior produto que já comprei", "Este produto é ok", 
           "Eu não compraria este produto novamente", "Este produto é incrível", 
           "Eu não gostei deste produto", "Eu compraria este produto novamente", 
           "Este produto é aceitável", "Eu gosto deste produto", "Este produto é terrível",
           "Este produto é mediano", "Eu não recomendaria este produto", 
           "Este produto é fantástico", "Eu odeio este produto", 
           "Eu poderia comprar este produto novamente"]

sentiments = ["positive", "negative", "neutral", "negative", "positive", 
              "negative", "positive", "neutral", "positive", "negative", 
              "neutral", "negative", "positive", "negative", "positive"]

# Pré-processamento de dados
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess(review):
    # Tokenização
    words = word_tokenize(review)
    # Remoção de palavras irrelevantes e lematização
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return words

processed_reviews = [preprocess(review) for review in reviews]

# Divisão dos dados em treinamento e validação
train_reviews, validation_reviews, train_sentiments, validation_sentiments = train_test_split(processed_reviews, sentiments, test_size=0.75, stratify=sentiments)

# Construção do modelo Word2Vec
w2v_model = Word2Vec(train_reviews, vector_size=100, window=5, min_count=1, workers=4)
w2v_model.train(train_reviews, total_examples=w2v_model.corpus_count, epochs=10)

# Transformação dos dados de treinamento e validação em vetores usando o modelo Word2Vec
train_vectors = np.array([np.mean([w2v_model.wv[word] for word in review if word in w2v_model.wv.key_to_index], axis=0) for review in train_reviews])
validation_vectors = np.array([np.mean([w2v_model.wv[word] for word in review if word in w2v_model.wv.key_to_index], axis=0) for review in validation_reviews])

# Classificação usando MLP
mlp = MLPClassifier(max_iter=1000, learning_rate_init=0.001)
mlp.fit(train_vectors, train_sentiments)
w2v_predictions = mlp.predict(validation_vectors)

# Construção do modelo Bag of Words com transformação TFIDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform([' '.join(review) for review in train_reviews])
validation_vectors = vectorizer.transform([' '.join(review) for review in validation_reviews])

# Classificação usando MLP
mlp = MLPClassifier(max_iter=1000, learning_rate_init=0.001)
mlp.fit(train_vectors, train_sentiments)
bow_predictions = mlp.predict(validation_vectors)

# Comparação dos modelos
print(f'Acurácia do Word2Vec: {accuracy_score(validation_sentiments, w2v_predictions)}')
print(f'Acurácia do Bag of Words: {accuracy_score(validation_sentiments, bow_predictions)}')

# Aplicação baseada em Word2Vec para reescrita de frases
def rewrite_sentence(sentence):
    words = preprocess(sentence)
    new_words = []
    for word in words:
        # Encontre a palavra mais similar no modelo Word2Vec
        try:
            similar_words = w2v_model.wv.most_similar(word)
            # Substitua a palavra pela palavra mais similar
            new_words.append(similar_words[0][0])
        except KeyError:
            new_words.append(word)
    return ' '.join(new_words)

sentence = "Eu adoro este produto"
new_sentence = rewrite_sentence(sentence)
print(new_sentence)