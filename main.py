from Read import Read
from Vectorize import Vectorize
from Encrypt import PaillierEncryptor
from Optimise import Optimise
from ModelController import ModelController

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
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

    enc_svd_train_fvm_path ="data/resulting/enc_svd_train_fvm(45comp).pkl"
    enc_svd_test_fvm_path  ="data/resulting/enc_svd_test_fvm(45comp).pkl"


    nmf_train_fvm_path = "data/intermediary/nmf_train_fvm(45comp).pkl"
    nmf_test_fvm_path  = "data/intermediary/nmf_test_fvm(45comp).pkl"

    enc_nmf_train_fvm_path ="data/resulting/enc_nmf_train_fvm(45comp).pkl"
    enc_nmf_test_fvm_path  ="data/resulting/enc_nmf_test_fvm(45comp).pkl"

    pca_train_fvm_path = "data/intermediary/pca_train_fvm(45comp).pkl"
    pca_test_fvm_path  = "data/intermediary/pca_test_fvm(45comp).pkl"

    enc_pca_train_fvm_path ="data/resulting/enc_pca_train_fvm(45comp).pkl"
    enc_pca_test_fvm_path  ="data/resulting/enc_pca_test_fvm(45comp).pkl"

    # Step 1: Read data
    train_data_texts  = Read.read_jsonl_text(training_file)
    train_data_labels = Read.read_jsonl_label(training_file)

    test_data_texts  = Read.read_jsonl_text(test_file)
    test_data_labels = Read.read_jsonl_label(test_file)
    
   
    # Step 2: Vectorize data
    print()
    X_train, vectorize = Vectorize.vectorize_data(train_data_texts)
    X_test = vectorize.transform(test_data_texts)
    print("X_train shape:", X_train.shape)
    print("X_train type:", type(X_train))
    print()

    # always compute the FVMs, it doesnt take too long 
    print("Computing FVMs...")
    X_train, vectorizer = Vectorize.vectorize_data(train_data_texts)
    X_test = vectorizer.transform(test_data_texts)
    save_matrix(X_train, train_fvm_path)
    save_matrix(X_test, test_fvm_path)

    print("Sample Train Row(X_train[0]):")
    print(X_train[0])
    print()
    print()

    print("X_train shape:", X_train.shape)
    print("X_train type:", type(X_train))
    print()
    print()


    # Step 3: Apply dimensionality reduction   
                #svd
                    # if os.path.exists(svd_train_fvm_path) and os.path.exists(svd_test_fvm_path):
                    #     print("Loading precomputed reduced matrices...")
                    #     X_train_reduced = load_matrix(svd_train_fvm_path)
                    #     X_test_reduced = load_matrix(svd_test_fvm_path)
                    # else:
                    #     print("Applying TruncatedSVD to reduce dimensionality...")
                    #     X_train_reduced, svd_model = Optimise.apply_truncated_svd(X_train, n_components=30)
                    #     X_test_reduced = svd_model.transform(X_test)
                    #     save_matrix(X_train_reduced, svd_train_fvm_path)
                    #     save_matrix(X_test_reduced, svd_test_fvm_path)
                    #     print("calculated the reduced matrices")



                    # if os.path.exists(nmf_train_fvm_path) and os.path.exists(nmf_test_fvm_path):
                    #     print("Loading precomputed reduced matrices...")
                    #     X_train_reduced = load_matrix(nmf_train_fvm_path)
                    #     X_test_reduced = load_matrix(nmf_test_fvm_path)
                    # else:
                    #     print("Applying NMF to reduce dimensionality...")
                    #     X_train_reduced, nmf_model = Optimise.apply_nmf(X_train, n_components=45)
                    #     X_test_reduced = nmf_model.transform(X_test)
                    #     save_matrix(X_train_reduced, nmf_train_fvm_path)
                    #     save_matrix(X_test_reduced, nmf_test_fvm_path)
                    #     print("calculated the reduced matrices")

    if os.path.exists(pca_train_fvm_path) and os.path.exists(pca_test_fvm_path):
        print("Loading precomputed reduced matrices...")
        X_train_reduced = load_matrix(pca_train_fvm_path)
        X_test_reduced = load_matrix(pca_test_fvm_path)
    else:
        print("Applying NMF to reduce dimensionality...")
        X_train_reduced, pca_model = Optimise.apply_pca(X_train, n_components=45)
        X_test_reduced = pca_model.transform(X_test)
        save_matrix(X_train_reduced, pca_train_fvm_path)
        save_matrix(X_test_reduced, pca_test_fvm_path)
        print("calculated the reduced matrices")

    print(f"Reduced Train Matrix Shape: {X_train_reduced.shape}")
    print(f"Reduced Test Matrix Shape: {X_test_reduced.shape}")
    print()


    # convert matrix to sparse representation
        # X_train_reduced = csr_matrix(X_train_reduced)
        # X_test_reduced  = csr_matrix(X_test_reduced)

        # print(f"Reduced Sparse Train Matrix Shape: {X_train_reduced.shape}")
        # print(f"Reduced Sparse Test Matrix Shape: {X_test_reduced.shape}")
        # print()


    # Step 4: Encrypt reduced data

    if os.path.exists(enc_pca_train_fvm_path) and os.path.exists(enc_pca_test_fvm_path):
        print("Loading precomputed encrypted FVMs...")
        X_train_encrypted = load_matrix(enc_pca_train_fvm_path)
        X_test_encrypted = load_matrix(enc_pca_test_fvm_path)
    else:
        encryptor = PaillierEncryptor()

        print("Encrypting reduced test data...")
        X_test_encrypted = encryptor.encrypt_feature_matrix(X_test_reduced)
        print("Test is encrypted, smaller but still cool")
        print()


        print("Encrypting FVMs...")
        print("Encrypting reduced training data...")
        X_train_encrypted = encryptor.encrypt_feature_matrix(X_train_reduced)
        print("Train is encrypted, phew that was a big file")

       

        save_matrix(X_train_encrypted, enc_pca_train_fvm_path)
        save_matrix(X_test_encrypted, enc_pca_test_fvm_path)

        print("Encryption complete.")
        print()







    # this is where we can confirm encryption 
    print("Sample Encrypted Train Row:", X_train_encrypted[0])
    print()
    print("Sample Encrypted Test Row:", X_test_encrypted[0])
    print()


    # Step 5: Train model
    model = LogisticRegression()
    #model = MultinomialNB()

    model_controller = ModelController(model)
    model_controller.train(X_train_encrypted, train_data_labels)


    # Step 6: Evaluate model
    model_controller.evaluatePerformance(X_test_encrypted, test_data_labels)


if __name__ == "__main__":
    main()
