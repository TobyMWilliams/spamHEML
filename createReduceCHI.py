import sys
import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from tqdm import tqdm 
import time



from sklearn.feature_selection import SelectKBest, chi2
from Optimise import Optimise, approximate_retained_variance
from Read import Read




def global_scale(train, test):
    #attempting to preserve non-negativity and vector directions
    start_scale = time.time()
    all_data = np.vstack([train, test])
    scaler = MinMaxScaler(feature_range=(0, 1))

    all_scaled = scaler.fit_transform(all_data)

    train_scaled = all_scaled[:len(train)]
    test_scaled = all_scaled[len(train):]
    scale_time = time.time() - start_scale
    print(f"time taken to reduce GloVe: {scale_time:.5f}")

    return train_scaled, test_scaled


def reduce_with_chi2(train_path, test_path, label_path, k=5000):
    print(f"\nReducing: {train_path}")

    # Load matrices and labels
    X_train = load_matrix(train_path)
    X_test = load_matrix(test_path)
    y_train = label_path
    # Ensure dense
    # Safely handle sparse or dense inputs
    if not isinstance(X_train, np.ndarray) and hasattr(X_train, "tocsr"):
        X_train = X_train.tocsr()
    if not isinstance(X_test, np.ndarray) and hasattr(X_test, "tocsr"):
        X_test = X_test.tocsr()

    # Scale globally (ensures non-negativity)
    # X_train_scaled, X_test_scaled = global_scale(X_train, X_test)
    # Apply chi2 using Optimise

    X_train_reduced, selector = Optimise.apply_chi2(X_train, y_train, k)

    start_test = time.time()
    X_test_reduced = selector.transform(X_test)
    test_time = time.time() - start_test
    print (f"ChiReduction to 5000 for {test_path}:  {test_time:.5f} seconds")
    

    # Save reduced outputs
    # save_matrix(X_train_reduced, out_train_path)
    # save_matrix(X_test_reduced, out_test_path)

def reduce_with_chi1(train_path, test_path, label_path, out_train_path, out_test_path, k=5000):
    print(f"\nReducing: {train_path}")

    # Load matrices and labels
    X_train = load_matrix(train_path)
    X_test = load_matrix(test_path)
    y_train = label_path
    # Ensure dense
    # Leave as sparse unless absolutely needed
    if not isinstance(X_train, np.ndarray) and hasattr(X_train, "tocsr"):
        X_train = X_train.tocsr()
    if not isinstance(X_test, np.ndarray) and hasattr(X_test, "tocsr"):
        X_test = X_test.tocsr()


    # Scale globally (ensures non-negativity)
    X_train_reduced, X_test_reduced = global_scale(X_train, X_test)


    # Save reduced outputs
    # save_matrix(X_train_reduced, out_train_path)
    # save_matrix(X_test_reduced, out_test_path)

def save_matrix(matrix, filepath):
    """Save the matrix to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(matrix, f)


def load_matrix(filepath):
    """Load the matrix from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    




def main():


    in_paths = {
        #sparse data
        "bow_train": "data/featureData/Bow_train.pkl",
        "bow_test": "data/featureData/Bow_test.pkl",

        "tfidf_train": "data/featureData/TF_IDF_train.pkl",
        "tfidf_test": "data/featureData/TF_IDF_test.pkl",

        "trigram_train": "data/featureData/triGram_train.pkl",
        "trigram_test": "data/featureData/triGram_test.pkl",

        "bigram_train": "data/featureData/biGram_train.pkl",
        "bigram_test": "data/featureData/biGram_test.pkl",

        # dense data
        "embed_train": "data/featureData/Embedding_train.pkl",
        "embed_test": "data/featureData/Embedding_test.pkl",
    
        # "sentiment_train": "data/featureData/Sentiment_train.pkl",
        # "sentiment_test": "data/featureData/Sentiment_test.pkl",
    }

    
    
    training_file = "data/train.jsonl"
    label_path = Read.read_jsonl_label(training_file)

        
    for name in ["bow", "bigram", "trigram", "tfidf"]:
        print(f"[{name.upper()}] Rebuilding chi2-reduced matrices...")
        
        
        # start_train = time.time()
        reduce_with_chi2(
            in_paths[f"{name}_train"],
            in_paths[f"{name}_test"],
            label_path,
            # f"data/featureData/CHIreduced/{name}_chi_train.pkl",
            # f"data/featureData/CHIreduced/{name}_chi_test.pkl"
        )
        # train_time = time.time() - start_train
        # print (f"ChiReduction to 5000 for {name}:  {train_time:.5f} seconds")


    embed = "embed"
    start_scale = time.time()
    reduce_with_chi1(in_paths[f"{embed}_train"], in_paths[f"{embed}_test"],
                    label_path,
                    f"data/featureData/CHIreduced/{embed}_chi_train.pkl",
                    f"data/featureData/CHIreduced/{embed}_chi_test.pkl"
                     )
    scale_time = time.time() - start_scale
    print(f"time taken to reduce GloVe: {scale_time:.5f}")
if __name__ == "__main__":
    main()