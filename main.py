from Read import Read
from Vectorize import Vectorize
from Encrypt import CKKS_Encryptor
from Optimise import Optimise
from ModelController import ModelController

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os



def save_matrix(matrix, filepath):
    """Save the matrix to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(matrix, f)


def load_matrix(filepath):
    """Load the matrix from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def main():

    # File paths
    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"


    train_fvm_path = "data/intermediary/train_fvm(45comp).pkl"
    test_fvm_path  = "data/intermediary/test_fvm(45comp).pkl"



    svd_train_fvm_path = "data/intermediary/svd_train_fvm(30comp).pkl"
    svd_test_fvm_path  = "data/intermediary/svd_test_fvm(30comp).pkl"

    pca_train_fvm_path = "data/intermediary/pca_train_fvm(45comp).pkl"
    pca_test_fvm_path  = "data/intermediary/pca_test_fvm(45comp).pkl"

    nmf_train_fvm_path = "data/intermediary/nmf_train_fvm(45comp).pkl"
    nmf_test_fvm_path  = "data/intermediary/nmf_test_fvm(45comp).pkl"



    enc_nmf_train_fvm_path ="data/intermediary/resulting/enc_nmf_train_fvm(45comp).pkl"
    enc_nmf_test_fvm_path  ="data/intermediary/resulting/enc_nmf_test_fvm(45comp).pkl"

    enc_svd_train_fvm_path ="data/intermediary/resulting/enc_svd_train_fvm(45comp).pkl"
    enc_svd_test_fvm_path  ="data/intermediary/resulting/enc_svd_test_fvm(45comp).pkl"

    enc_pca_train_fvm_path ="data/intermediary/resulting/enc_pca_train_fvm(45comp).pkl"
    enc_pca_test_fvm_path  ="data/intermediary/resulting/enc_pca_test_fvm(45comp).pkl"

    # Step 1: Read data
    train_data_texts  = Read.read_jsonl_text(training_file)
    train_data_labels = Read.read_jsonl_label(training_file)

    test_data_texts  = Read.read_jsonl_text(test_file)
    test_data_labels = Read.read_jsonl_label(test_file)
    
   


    # always compute the FVMs, it doesnt take too long 
    print("Computing FVMs...")
    X_train, vectorizer = Vectorize.vectorize_data(train_data_texts)
    X_test = vectorizer.transform(test_data_texts)
    save_matrix(X_train, train_fvm_path)
    save_matrix(X_test, test_fvm_path)
    print()

    # apply PCA for dim red
    print("Applying PCA to reduce dimensionality...")
    X_train_reduced, pca_model = Optimise.apply_pca(X_train, n_components=45)
    X_test_reduced = pca_model.transform(X_test)
    save_matrix(X_train_reduced, pca_train_fvm_path)
    save_matrix(X_test_reduced, pca_test_fvm_path)
    print("calculated the reduced matrices")

    print(f"Reduced Train Matrix Shape: {X_train_reduced.shape}")
    print(f"Reduced Test Matrix Shape: {X_test_reduced.shape}")
    print()


  


    encryptor = CKKS_Encryptor()
    encryption_context = encryptor.get_encryption_context()

    print("Encrypting reduced test data...")
    X_test_encrypted = encryptor.encrypt_feature_matrix(X_test_reduced)
    print()



    # Train model
    model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=42)

    model_controller = ModelController(model)  
    model_controller.train(X_train_reduced, train_data_labels)



    # todo - this is where I should encrypt the model weights for full security




    # Evaluate model
    model_controller.evaluatePerformance(X_test_encrypted, test_data_labels)


if __name__ == "__main__":
    main()
