import json
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize




# housing all the vectorization logic 

class TFIDFVectorizer:
    @staticmethod
    def vectorize_data(data, dense=False):
        """TF-IDF vectorizer"""
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        X = vectorizer.fit_transform(data)
        return (X.toarray() if dense else X), vectorizer


class BoWVectorizer:
    @staticmethod
    def vectorize_data(data):
        """Classic Bag-of-Words vectorizer."""
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(data), vectorizer


class NGramVectorizer:
    @staticmethod
    def vectorize_data(data, ngram_range):
        """N-gram vectorizer (used for bigrams and trigrams)."""
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        return vectorizer.fit_transform(data), vectorizer



# dense representation 
class EmbeddingVectorizer:
    @staticmethod
    def load_embeddings(filepath):
        """Load pretrained embeddings from file (e.g., GloVe format)."""
        embeddings = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    @staticmethod
    def vectorize_data(data, embeddings_dict, dim=300):
        vectors = []
        for text in data:
            words = text.split()
            word_vectors = [embeddings_dict[word] for word in words if word in embeddings_dict]
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(dim)
            vectors.append(doc_vector)
        return np.array(vectors)


