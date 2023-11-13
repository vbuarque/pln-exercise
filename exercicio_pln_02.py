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