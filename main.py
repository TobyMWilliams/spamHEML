from Read import Read
from Vectorize import TFIDFVectorizer
from Encrypt import CKKS_Encryptor
from Optimise import Optimise
from SVMcontroller import SVMController
from SGDcontroller import SGDcontroller
from LogRegController import LogisticRegressionController
from GBcontroller import GBcontroller
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
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score





def main():

    # File paths
    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"

    # Step 1: Read data
    train_data_texts  = Read.read_jsonl_text(training_file)
    train_data_labels = Read.read_jsonl_label(training_file)

    test_data_texts  = Read.read_jsonl_text(test_file)
    test_data_labels = Read.read_jsonl_label(test_file)

    # Step 2: Vectorize data
    X_train, tfidf_vec = TFIDFVectorizer.vectorize_data(train_data_texts)
    X_test = tfidf_vec.transform(test_data_texts)


    # Step 3: Reduce dimensionality (CHI) 
    X_train_reduced_chi, selector = Optimise.apply_chi2(X_train, train_data_labels, k = 5000)
    X_test_reduced = selector.transform(X_test)

    # Step 4: Reduce dimensionality (SVD)
    X_train_reduced_svd, model  = Optimise.apply_truncated_svd(X_train_reduced_chi, k=45)
    X_test_reduced_svd = model.transform(X_test_reduced)


     # Step 5: train model (unencrypted)
    model_controller = SVMController()
    model_controller.train(X_train_reduced_svd, train_data_labels)
    
     # Step 6: Encrypt test data
    encryptor = CKKS_Encryptor()
    X_test_encrypted = encryptor.encrypt_feature_matrix(X_test_reduced_svd)

    # Step 7: Run inference
    encrypted_inference = model_controller.makeInference(X_test_encrypted)
    client = ClientSide() 
    cm = client.decrypt_and_classify_LR(  # reuse sigmoid+threshold logic
        encrypted_inference=encrypted_inference,
        threshold=0.587,
        testLabels=test_data_labels
    )




if __name__ == "__main__":
    main()