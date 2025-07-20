from Read import Read
from Vectorize import TFIDFVectorizer, BoWVectorizer, NGramVectorizer, EmbeddingVectorizer, SentimentVectorizer

import pickle
import time


GloVeEmbeddingDB = "data/glove.6B.300d.txt"


def save_matrix(matrix, filepath):
    """Save the matrix to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(matrix, f)


def load_matrix(filepath):
    """Load the matrix from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def main():

    # in File paths
    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"

    # out file paths
    out_paths = {
        "bow_train": "data/featureData/Bow_train.pkl",
        "bow_test": "data/featureData/Bow_test.pkl",

        "tfidf_train": "data/featureData/TF_IDF_train.pkl",
        "tfidf_test": "data/featureData/TF_IDF_test.pkl",

        "trigram_train": "data/featureData/triGram_train.pkl",
        "trigram_test": "data/featureData/triGram_test.pkl",

        "bigram_train": "data/featureData/biGram_train.pkl",
        "bigram_test": "data/featureData/biGram_test.pkl",

        "embed_train": "data/featureData/Embedding_train.pkl",
        "embed_test": "data/featureData/Embedding_test.pkl",
    
    
    }
    print()
    train_texts = Read.read_jsonl_text(training_file)
    test_texts  = Read.read_jsonl_text(test_file)

    print("\n[1] Extracting Bag-of-Words...")
    start_train = time.time()
    X_train, bow_vec = BoWVectorizer.vectorize_data(train_texts)
    X_test = bow_vec.transform(test_texts)
    train_time = time.time() - start_train
    print (f"Extracting BoW:  {train_time:.5f} seconds")
    print()
    save_matrix(X_train, out_paths["bow_train"])
    save_matrix(X_test, out_paths["bow_test"])


    print("\n[2] Extracting TF-IDF...")
    X_train, tfidf_vec = TFIDFVectorizer.vectorize_data(train_texts)
    start_train = time.time()
    X_test = tfidf_vec.transform(test_texts)
    train_time = time.time() - start_train
    print (f"Extracting TF-IDF:  {train_time:.5f} seconds")
    print()
    save_matrix(X_train, out_paths["tfidf_train"])
    save_matrix(X_test, out_paths["tfidf_test"])


    print("\n[3] Extracting bi-grams...")
    X_train, ngram_vec = NGramVectorizer.vectorize_data(train_texts, ngram_range=(2, 2))
    start_train = time.time()
    X_test = ngram_vec.transform(test_texts)
    train_time = time.time() - start_train
    print (f"Extracting bi-grams:  {train_time:.5f} seconds")
    print()
    save_matrix(X_train, out_paths["bigram_train"])
    save_matrix(X_test, out_paths["bigram_test"])

    print("\n[4] Extracting trigrams...")
    X_train, ngram_vec = NGramVectorizer.vectorize_data(train_texts, ngram_range=(3,3))
    start_train = time.time()
    X_test = ngram_vec.transform(test_texts)
    train_time = time.time() - start_train
    print (f"Extracting trigrams:  {train_time:.5f} seconds")
    print()
    save_matrix(X_train, out_paths["trigram_train"])
    save_matrix(X_test, out_paths["trigram_test"])


    print("\n[5] Extracting Word Embedding Averages(GloVe)...")
    embeddings = EmbeddingVectorizer.load_embeddings(GloVeEmbeddingDB)
    X_train = EmbeddingVectorizer.vectorize_data(train_texts, embeddings)
    #stat_time = time.time()
    embeddings = EmbeddingVectorizer.load_embeddings(GloVeEmbeddingDB)
    start_train = time.time()
    X_test = EmbeddingVectorizer.vectorize_data(test_texts, embeddings)
    train_time = time.time() - start_train
    print (f"Extracting GloVe:  {train_time:.5f} seconds")
    save_matrix(X_train, out_paths["embed_train"])
    save_matrix(X_test, out_paths["embed_test"])



    print("\n Feature extraction and saving complete.")


if __name__ == "__main__":
    main()
