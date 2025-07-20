
from Read import Read
from Vectorize import TFIDFVectorizer
from Encrypt import CKKS_Encryptor
from Optimise import Optimise
from SGDcontroller import SGDcontroller
from GBcontroller import GBcontroller
from LogRegController import LogisticRegressionController
from SVMcontroller import SVMController
from clientSide import ClientSide

# imported models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier



# imported libraries
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import time


def save_matrix(matrix, filepath):
    """Save the matrix to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(matrix, f)

def load_matrix(filepath):
    """Load the matrix from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else X




def run_svm_inference(X_train, y_train, X_test_encrypted, y_test, label):
    print(f"\n=== Running SVM Inference for {label} ===")

    model_controller = SVMController()

    model_controller.train(X_train, y_train)

    client = ClientSide()

    start_test_svm = time.time()
    encrypted_logits = model_controller.makeInference(X_test_encrypted)

     # reuse sigmoid+threshold logic
    cm = client.decrypt_and_classify_LR( 
        encrypted_inference=encrypted_logits,
        threshold=0.587,
        testLabels=y_test
    )
    test_time_hinge = time.time() - start_test_svm
    print(f"Time taken to test and announce SVM: {test_time_hinge:.5f} seconds")

    



def main():
    # File paths
    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"

    train_data_labels = Read.read_jsonl_label(training_file)
    test_data_labels = Read.read_jsonl_label(test_file)

    train_texts = Read.read_jsonl_text(training_file)
    test_texts  = Read.read_jsonl_text(test_file)


    start_scale = time.time()

    print("\n Extracting TF-IDF...")
    X_train, tfidf_vec = TFIDFVectorizer.vectorize_data(train_texts)
    X_test = tfidf_vec.transform(test_texts)

    # preprocess the data with chi2
    X_train, model_chi = Optimise.apply_chi2(X_train, train_data_labels, 5000)
    X_test = model_chi.transform(X_test)
                
    #reduduce dimensions with PCA
    X_train, model_svd = Optimise.apply_truncated_svd(X_train, 45)
    scale_time = time.time() - start_scale
    print (f"Scaling to 45 using SVD: {training_file}:  {scale_time:.5f} seconds")
    
    start_transform = time.time()

    X_test = model_svd.transform(X_test)

    encryptor = CKKS_Encryptor()
    X_test_encrypted = encryptor.encrypt_feature_matrix(X_test)
    scale_time4 = time.time() - start_transform
    print (f"[SVD] transform and encrypt {test_file}:  {scale_time4:.5f} seconds")

    run_svm_inference(
            X_train = X_train,
            y_train = train_data_labels,
            X_test_encrypted = X_test_encrypted,
            y_test = test_data_labels,
            label = ("Best time complexity combination"),
        )



if __name__ == "__main__":
    main()
