import json
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorize:
    @staticmethod
    def vectorize_data(data):
        """Vectorize the given data."""
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(data), vectorizer